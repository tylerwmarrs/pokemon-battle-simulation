from multiprocessing import Pool, cpu_count
import os

import numpy as np
import pandas as pd

ROOT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)
DATA_DIR = os.path.join(ROOT_DIR, 'data')

# load in type modifiers, pokemon, etc
POKEMON_MOVES = pd.read_csv(os.path.join(DATA_DIR, 'processed', 'pokemon_moves_detailed.csv'))
POKEMON_STATS = pd.read_csv(os.path.join(DATA_DIR, 'processed', 'pokemon_stats.csv'))
TYPE_MODS = pd.read_csv(os.path.join(DATA_DIR, 'processed', 'type_modifiers.csv')).set_index('attack_type')
POKEMON_AVAIL = set(list(POKEMON_STATS['pokemon'].unique()))
VERBOSE = False
VERBOSE_COUNT = True
NUM_SIMULATIONS = 1000

class Move(object):
    """
    Encapsulates a move to apply for a given pokemon. It is used to keep track
    of the power points available, damage and move type.
    """
    def __init__(self):
        self.current_pp = None
    
    def __str__(self):
        out = []
        out.append('Name: {}'.format(self.name))
        out.append('Type: {}'.format(self.type))
        
        for k, v in self.__dict__.items():
            if k in ('name', 'type', 'url', 'pokemon') or v is None:
                continue
            
            out.append('{}: {}'.format(k, v))
        
        return '\n'.join(out)


class Pokemon(object):
    """
    Encapsulates the moves and stats of a Pokemon for battle. We also use it
    to keep track of the current hit points and moves.
    """
    def __init__(self, name):
        self.name = name.lower()
        self.types = []
        self.current_hp = None
        
        self.all_moves = []
        self.moves = []
        self.attack = None
        self.defense = None
        self.hp = None
        self.special_attack = None
        self.special_defense = None
        self.speed = None
        
        if self.name not in POKEMON_AVAIL:
            raise RuntimeError('{} is not available!'.format(self.name))
        
        self.__load_stats()
        self.__load_moves()
        self.pick_moves()
    
    def __load_stats(self):        
        query = POKEMON_STATS['pokemon'] == self.name
        if query.sum() != 1:
            raise RuntimeError('{} expecting 1 result for stats, got {}'.format(self.name, query.sum()))
        
        stats = POKEMON_STATS[query].iloc[0].to_dict()
        for k, v in stats.items():
            if k == 'types':
                setattr(self, k, set(v.split(',')))
            else:            
                setattr(self, k.replace('-', '_'), v)
    
    def __load_moves(self):
        query = POKEMON_MOVES['pokemon'] == self.name
        if query.sum() < 1:
            raise RuntimeError('{} has no moves!'.format(self.name))
        
        for idx, row in POKEMON_MOVES[query].drop_duplicates().iterrows():
            move = Move()
            dict_row = row.to_dict()
            for k, v in dict_row.items():
                val = v
                if isinstance(val, float) and np.isnan(val):
                    val = None
                
                if isinstance(val, str) and val.strip() == '':
                    val = None
                
                setattr(move, k.replace('move_', ''), val)
            
            self.all_moves.append(move)
    
    def pick_moves(self):
        # only pick damaging moves
        damage_moves = []
        for move in self.all_moves:
            if 'damage' in move.category:
                damage_moves.append(move)
        
        # enable fewer than 4 moves to be randomly chosen
        max_moves = 4
        if len(damage_moves) < max_moves:
            max_moves = len(damage_moves)                                
        
        self.moves = np.random.choice(damage_moves, max_moves, replace=False)
        
        self.has_moves = True
        if len(self.moves) < 1:
            self.has_moves = False
            
    def reset(self):
        self.current_hp = self.hp
        self.pick_moves()
        
        for move in self.moves:
            move.current_pp = move.pp
    
    
    def __str__(self):
        move_str = []
        for move in self.moves:
            move_str.append('{} - {}'.format(move.name, move.type))
            
        return """
        =================
        Pokemon: {}
        =================
        Types:         {}
        HP:            {}
        Speed:         {}
        Attack:        {}
        Defense:       {}
        Sp. Attack:    {}
        Sp. Defense:   {}
        =====
        Moves
        =====
        {}
        """.format(
            self.name.title(),
            ', '.join(self.types),
            self.hp,
            self.speed,
            self.attack,
            self.defense,
            self.special_attack,
            self.special_defense,
            '\n'.join(move_str),
        )

def is_critical_hit(base_speed, move_crit_rate):
    """
    TODO: amping abilities not applied - focus energy etc..
    this is bugged in Gen 1 - but we will just ignore it.
    """
    prob = base_speed / 512
    if move_crit_rate == 1:
        prob = base_speed / 64
    
    chance = np.random.rand()
    return chance <= prob


def calculate_damage(a, b, c, d, x, y, crit):
    """
    a = attacker's Level
    b = attacker's Attack or Special
    c = attack Power
    d = defender's Defense or Special
    x = same-Type attack bonus (1 or 1.5)
    y = Type modifiers (40, 20, 10, 5, 2.5, or 0)
    z = a random number between 217 and 255
    crit = true or false
    """
    z = np.random.choice(np.arange(217, 256))
    crit_mult = 2
    if crit:
        crit_mult = 4
    
    damage = np.floor(((((crit_mult * a) / 5) + 2) * b * c) / d)
    damage = np.floor(damage / 50) + 2
    damage = np.floor(damage * x)    
    damage = np.floor(damage * y)
    
    # This is how you could compute the minimum and maximum
    # damage that this ability will do. Only for reference.
    # min_damage = np.floor((damage * 217) / 255)
    # max_damage = np.floor((damage * 255) / 255)
    
    return np.floor((damage * z) / 255)


def apply_move(attacker, defender, move):
    """
    Applies a damaging move attacker->defender where the attacker and defenders
    are instances of Pokemon. The move is an instance of the move being
    applied.
    """
    # determine if it is a critical hit or not
    is_crit = is_critical_hit(attacker.speed, move.crit_rate)
    
    # determine if move applied is the same type as the pokemon or not
    # when it is the same, a 1.5x bonus is applied
    # STAB = same type attack bonus
    stab = 1
    if move.type in attacker.types:
        stab = 1.5
    
    # determine the move damage class to figure out attack/def stats to use
    attack = attacker.attack
    defense = defender.defense
    if move.damage_class == 'special':
        attack = attacker.special_attack
        defense = defender.special_defense
    
    # grab type modifier
    modifier = 1
    try:
        attack_type = move.type.title()
        for dtype in defender.types:
            modifier *= TYPE_MODS.loc[attack_type][dtype.title()]
    except:
        pass
    
    # NOTE: attacker level is hard coded to 10
    level = 10
    power = move.power
    if power is None:
        power = 1
    
    damage = 1
    if move.name == 'seismic-toss':
        damage = level
    else:
        damage = calculate_damage(level, attack, power, defense, stab, modifier, is_crit)
    
    # compute number of times to apply the move
    times_to_apply = 1
    if move.min_hits and move.max_hits:
        times_to_apply = np.random.choice(np.arange(move.min_hits, move.max_hits + 1))
    
    damage *= times_to_apply
    
    # apply damage to pokemon and reduce move pp
    defender.current_hp -= damage
    move.current_pp -= 1
    
    if VERBOSE:
        print('{} damaged {} with {} for {} hp'.format(
            attacker.name,
            defender.name,
            move.name,
            damage
        ))
        print('{} pp for {} is {}/{}'.format(attacker.name, move.name, move.current_pp, move.pp))
        print('{} hp is {}/{}'.format(defender.name, defender.current_hp, defender.hp))
    

def choose_move(pokemon):
    """
    Naive and exhaustive approach in choosing a move. It does not pick the most
    optimal move; only random. We also ensure that there is enough PP.
    """
    iters = 0
    move = None
    while move is None and iters < 100:
        move = np.random.choice(pokemon.moves)
        if move.current_pp < 1:
            move = None
        
        iters += 1
    
    return move
        
        
def battle(pokemon, pokemon_b):
    """
    Simulates a single battle against the two provided Pokemon instances. The
    battle is concluded when a pokemon loses all of their HP or both pokemon
    run out of PP for all of their moves.
    
    Pokemon are reset at the beginning of each battle. A reset consists of:
    
    1. Random damaging moves selected (up to 4)
    2. PP for moves being restored
    3. HP being reset
    
    We also randomly choose which pokemon attacks first with an equal chance.
    """
    stats = {
        'pokemon': pokemon.name,
        'pokemonb': pokemon_b.name,
        'moves': 0,
        'winner': None,
        'first_attack': None
    }
    
    pokemon.reset()
    pokemon_b.reset()
    
    start = np.random.choice(['a', 'b'])
    if start == 'a':
        stats['first_attack'] = pokemon.name
    else:
        stats['first_attack'] = pokemon_b.name
    
    while True:
        moves_exhausted = False
        
        if start == 'a':
            attacker = pokemon
            defender = pokemon_b
        else:
            attacker = pokemon_b
            defender = pokemon
        
        # starter attacks first
        stats['moves'] += 1
        move = choose_move(attacker)
        
        if move is not None:
            apply_move(attacker, defender, move)
        else:
            moves_exhausted = True
        
        if defender.current_hp <= 0:
            stats['winner'] = attacker.name
            break
        
        if start == 'a':
            attacker = pokemon_b
            defender = pokemon
        else:
            attacker = pokemon
            defender = pokemon_b
        
        # next pokemon attacks
        stats['moves'] += 1
        move = choose_move(attacker)
        
        if move is not None:
            apply_move(attacker, defender, move)
            moves_exhausted = False
        else:
            moves_exhausted = True
        
        if defender.current_hp <= 0:
            stats['winner'] = attacker.name
            break
        
        # handle case where all moves exhausted and no winner
        if moves_exhausted:
            stats['winner'] = None
            break
    
    return stats


def battle_many(opponents):
    """
    Parallelizable function that takes a tuple of pokemon names to battle.
    
    Ex:
    (pikachu, gastly)
    
    The two pokemon battle each other N number of times. The statistics
    are aggregated and returned.
    """
    stats = {
        'pokemon': [],
        'pokemonb': [],
        'avg_moves': [],
        'pokemon_wins': [],
        'pokemonb_wins': [],
        'ties': [],
    }
    
    pokemon = Pokemon(opponents[0])
    pokemon_b = Pokemon(opponents[1])
    print('{} vs {}'.format(pokemon.name.title(), pokemon_b.name.title()))
    
    battle_stats = {
        'a_wins': 0,
        'b_wins': 0,
        'ties': 0,
        'moves': [],
    }

    for _ in range(NUM_SIMULATIONS):
        result = battle(pokemon, pokemon_b)
        if result['winner'] == pokemon.name:
            battle_stats['a_wins'] += 1
        elif result['winner'] == pokemon_b.name:
            battle_stats['b_wins'] += 1
        else:
            battle_stats['ties'] += 1

        battle_stats['moves'].append(result['moves'])
    
    stats['pokemon'].append(pokemon.name)
    stats['pokemonb'].append(pokemon_b.name)
    stats['avg_moves'].append(np.array(battle_stats['moves']).mean())
    stats['pokemon_wins'].append(battle_stats['a_wins'])
    stats['pokemonb_wins'].append(battle_stats['b_wins'])
    stats['ties'].append(battle_stats['ties'])
    
    return pd.DataFrame(stats)

def main():
    # find pokemon that actually have damaging moves
    valid_pokemon = []
    for pokemon in POKEMON_AVAIL:
        p = Pokemon(pokemon)
        
        if p.has_moves:
            valid_pokemon.append(p.name)
        

    # construct pokemon matches as list of list where each sub-list contains
    # the two pokemon names that will battle
    battles = {}
    for i in valid_pokemon:
        for j in valid_pokemon:
            if i == j:
                continue
            
            opponents = [i, j]
            opponents.sort()
            battle_key = ','.join(opponents)
            battles[battle_key] = opponents
    
    matches = list(battles.values())
    
    # simulate the battles in parallel
    with Pool(cpu_count()) as pool:
        stats = pool.map(battle_many, matches)
    
    # output the results
    pd.concat(stats).to_csv(os.path.join(DATA_DIR, 'results', 'simulation_stats.csv'), index=False)
    
if __name__ == '__main__':
    main()