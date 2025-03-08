import pandas as pd
import sys
import numpy as np


def add_fractals(Data):
    """ Adds fractals to dataframe"""
    for i in range(2, len(Data) - 2):
        sides = [i-2,i-1,i+1,i+2]
        
        # Fractal Up (Bearish)
        Data.loc[i, 'fractal_up'] = 1 if all(Data.loc[i, 'high'] > Data.loc[j, 'high'] for j in sides) else 0
        
        # Fractal Down (Bullish)
        Data.loc[i, 'fractal_down'] = 1 if all(Data.loc[i, 'low'] < Data.loc[j, 'low'] for j in sides) else 0
    
    Data = Data.fillna(0)
                
    return Data

def fibo_retracement(low_swing, high_swing, fibo_level):
    diff = high_swing - low_swing
    
    # Calculate the retracement value for the given fibo_level
    retracement_value = high_swing - (diff * fibo_level)
    return retracement_value

def find_fractal_from(Data, ind, fractal_type, critical_level_1=None, first_fractal=None):
    """ Looks for fractal of type indicated starting from ind. If better fractal appears ind_A gets updated.
        ex: when looking for up fractal starting from A if a fractal lower than the one at ind_A appears, we update ind_A. 
        Returns tuple (status_1, status_2, ind_A, inb_B) with updated fractal locations. 
        status_1 is rejected if price crossed critical_level_1 and status_2 is rejected if price crossed 0.786 between A and B.
    """
    ind_A = ind_B = ind
    status_1, status_2 = 'accepted', 'accepted' # Status of both structures (used when checking level)
    lookout_frac = 'fractal_down' if fractal_type=='fractal_up' else 'fractal_up' # Fractal to look for in case ind has to be updated
    price_2 = Data.loc[ind, 'low'] if lookout_frac=='fractal_down' else Data.loc[ind, 'high']

    if first_fractal is not None:
        price_1 = Data.loc[first_fractal, 'low']
        critical_level_2 = fibo_retracement(price_1, price_2, 0.786)

    i = ind + 1
    for i in range(ind + 1, len(Data)):
        # When price goes below this level structure_1 is failed
        if critical_level_1 is not None and Data.loc[i, 'low'] < critical_level_1:
            status_1 = 'rejected'
            break
        # When price goes below this level structure_1 is failed
        if first_fractal is not None and Data.loc[i, 'low'] < critical_level_2:
            status_2 = 'rejected'
            break
        if Data.loc[i, lookout_frac]:
            # If price is more extreme than current, ind has to be updated
            if (lookout_frac=='fractal_down' and Data.loc[i, 'low'] < price_2) or (lookout_frac=='fractal_up' and Data.loc[i, 'high'] > price_2):
                ind_A = i
                if first_fractal is not None:
                    price_2 = Data.loc[ind, 'low'] if lookout_frac=='fractal_down' else Data.loc[ind, 'high']
                    critical_level_2 = fibo_retracement(price_1, price_2, 0.786)
        if Data.loc[i, fractal_type]:
            break
    
    return status_1, status_2, ind_A, i


def validate_structure(Data, ind_A, ind_B, critical_level_1=None):
    """ Validates structure ex:ABC drawing fibo retracements between A and B.
        We first locate down fractal after B to determine C. If higher fractal appears between B and C, B gets updated.
        When validating 2nd structure we have to set critical_level_1 to fibo 0.786 between A and B to keep checking price doesn't cross it
        If price goes below 0.786 at any point, struct is failed 
        If price at down fractal is above 0.535 it is rebounce
        If price at down fractal is between 0.696 and 0.535 it can be accepted

        Function returns a tuple with type of both structure, updated index of B and index of C
    """
    
    status_1, status_2, ind_B, ind_C = find_fractal_from(Data, ind_B, 'fractal_down',
                                                         critical_level_1=critical_level_1, first_fractal=ind_A)
    
    if status_1=='rejected' or status_2=='rejected': return status_1, status_2, ind_B, ind_C
    if ind_C >= len(Data): return None, None, ind_B, ind_C

    A = Data.loc[ind_A, 'low']
    B = Data.loc[ind_B, 'high'] 
    low_wick = min(Data.loc[ind_C, 'close'], Data.loc[ind_C, 'open']) # Get lower wick of point C
    
    # If whole body of candle goes above 0.535 it is rebounce
    if  low_wick >= fibo_retracement(A, B, 0.535): return status_1, 'rebounce', ind_B, ind_C
    else: return status_1, 'accepted', ind_B, ind_C


def find_fractal_rebounce(Data, ind_A, ind_B, critical_level_1=None):
    """ Finds next higher fractal after ind_B and updates ind_A if lower down fractal is found while 
        checking that price doesn't cross critical_level_1.
        Returns tuple with structure status, location of A and B
    """
    status = 'accepted'
    higher_wick = max(Data.loc[ind_B, 'close'], Data.loc[ind_B, 'open'])
    i = ind_B + 1
    
    for i in range(ind_B + 1, len(Data)):
        # When price goes below this level structure_1 is failed
        if critical_level_1 is not None and Data.loc[i, 'low'] < critical_level_1:
            status = 'rejected'
            break
       
        # If found lower fractal than A, ind_A has to be updated
        if Data.loc[i, 'fractal_down'] and Data.loc[i, 'low'] < Data.loc[ind_A, 'low']: ind_A = i
        
        # When up fractal higher than wick appears we found ind_B
        if Data.loc[i, 'fractal_up'] and Data.loc[i, 'high'] >= higher_wick: break
    
    return status, ind_A, i


def relocate_fractal(Data, ind_A, ind_B, critical_level_1=None):
    """ As long as structure is of type rebounce it relocates point B to next higher fractal found in Data, starting iterations from ind_B.
        The index where down_fractal A was found can also get updated. Function validates structure with updated locations.
    """
    ind_C = ind_B
    struct_type_1, struct_type_2 = 'accepted', 'rebounce'
    
    while ind_B < len(Data) and struct_type_2 == 'rebounce':
        struct_type_1, ind_A, ind_B = find_fractal_rebounce(Data, ind_A, ind_B, critical_level_1=critical_level_1)
        if struct_type_1 == 'rejected': return struct_type_1, struct_type_2, ind_A, ind_B, ind_C
        else: struct_type_1, struct_type_2, ind_B, ind_C = validate_structure(Data, ind_A, ind_B, critical_level_1=critical_level_1)
    
    return struct_type_1, struct_type_2, ind_A, ind_B, ind_C


def find_entry_from(Data, start):
    """ Executes entry strategy given starting index
        Returns list of indexes [A, B, C, D, E, F] if entry was found, otherwise empty list
    """
    ind_A = start

    # After "A" when an up fractal appears, control point "B" can be placed on the top wick.
    _, _, ind_A, ind_B = find_fractal_from(Data, ind_A, 'fractal_up')
    if ind_B >= len(Data): return []

    # Validate first structure
    _, struct_type, ind_B, ind_C = validate_structure(Data, ind_A, ind_B)

    # When structure is rebounce we have to relocate "B" to next higher fractal
    if struct_type == 'rebounce': 
        _, struct_type, ind_A, ind_B, ind_C = relocate_fractal(Data, ind_A, ind_B)

    # We reached end of dataframe
    if ind_C >= len(Data) or struct_type == 'rejected': return []

    A = Data.loc[ind_A, 'low']
    B = Data.loc[ind_B, 'high']
    fail_level = fibo_retracement(A, B, 0.786)

    # After "C" when an up fractal appears, control point "D" is placed
    struct_type_1, _, ind_C, ind_D = find_fractal_from(Data, ind_C, 'fractal_up', critical_level_1=fail_level)
    if ind_D >= len(Data) or struct_type_1 == 'rejected': return []
        
    # Check if it is rebounce
    if Data.loc[ind_C, 'low'] > fibo_retracement(Data.loc[ind_A, 'low'], Data.loc[ind_B, 'high'], 0.535):
        _, struct_type_1, ind_A, ind_B, ind_C = relocate_fractal(Data, ind_A, ind_B)
        if ind_C >= len(Data) or struct_type_1 == 'rejected': return []

    # C was accepted, it is closest point to level 0.618 in lower wick
    struct_type_1, struct_type_2, ind_D, ind_E = validate_structure(Data, ind_C, ind_D, critical_level_1=fail_level)
    
    while struct_type_1 == 'accepted' and (struct_type_2 == 'rebounce' or struct_type_2 == 'rejected') and ind_E <= len(Data):
        if struct_type_2 == 'rebounce':
            struct_type_1, struct_type_2, ind_C, ind_D, ind_E = relocate_fractal(Data, ind_C, ind_D, critical_level_1=fail_level)
            
            # Check if struct_1 turned to rebounce, in that case revalidate both
            if Data.loc[ind_C, 'low'] > fibo_retracement(Data.loc[ind_A, 'low'], Data.loc[ind_B, 'high'], 0.535):
                _, struct_type_1, ind_A, ind_B, ind_C = relocate_fractal(Data, ind_A, ind_B)
                if ind_C >= len(Data) or struct_type_1 == 'rejected': return []
                fail_level = fibo_retracement(Data.loc[ind_A, 'low'], Data.loc[ind_B, 'high'], 0.786)
                struct_type_1, struct_type_2, ind_C, ind_D = find_fractal_from(Data, ind_C, 'fractal_up', critical_level_1=fail_level)
                if ind_D >= len(Data) or struct_type_1 == 'rejected': return []
                struct_type_1, struct_type_2, ind_D, ind_E = validate_structure(Data, ind_C, ind_D, critical_level_1=fail_level)
                if ind_E >= len(Data) or struct_type_1 == 'rejected': return []
                    
        # Case where "ABCD" were marked but "E" is not accepted
        if struct_type_2 == 'rejected':
            # Revalidate first struct with new point B
            if Data.loc[ind_D, 'high'] > Data.loc[ind_B, 'high']:
                ind_B = ind_D
                fail_level = fibo_retracement(Data.loc[ind_A, 'low'], Data.loc[ind_B, 'high'], 0.786)
                struct_type_1, _, ind_B, ind_C = validate_structure(Data, ind_A, ind_B, critical_level_1=fail_level)
            
                # Check if it is rebounce
                if Data.loc[ind_C, 'low'] > fibo_retracement(Data.loc[ind_A, 'low'], Data.loc[ind_B, 'high'], 0.535):
                    fail_level = fibo_retracement(Data.loc[ind_A, 'low'], Data.loc[ind_B, 'high'], 0.786)
                    struct_type_1, _, ind_A, ind_B, ind_C = relocate_fractal(Data, ind_A, ind_B, critical_level_1=fail_level)
                if ind_C >= len(Data) or struct_type_1 == 'rejected': return []

                # Revalidate second struct
                fail_level = fibo_retracement(Data.loc[ind_A, 'low'], Data.loc[ind_B, 'high'], 0.786)
                struct_type_1, struct_type_2, ind_C, ind_D = find_fractal_from(Data, ind_C, 'fractal_up', critical_level_1=fail_level)
                if ind_D >= len(Data) or struct_type_1 == 'rejected': return []
                struct_type_1, struct_type_2, ind_D, ind_E = validate_structure(Data, ind_C, ind_D, critical_level_1=fail_level)
            
            # Same as rebounce of second structure, we have to relocate C, D
            else: struct_type_2 = 'rebounce'

    if struct_type_1 == 'rejected' or ind_E >= len(Data): return []

    A, B = Data.loc[ind_A, 'low'], Data.loc[ind_B, 'high']
    low_wick = min(Data.loc[ind_C, 'close'], Data.loc[ind_C, 'open'])
    C = np.clip(fibo_retracement(A, B, 0.618), Data.loc[ind_C, 'low'], low_wick)
    D = Data.loc[ind_D, 'high']
    fail_level_1 = fibo_retracement(A, B, 0.786)
    fail_level_2 = fibo_retracement(C, D, 0.786)
    fail_level = max(fail_level_1, fail_level_2)
    fibo_CD = fibo_retracement(C, D, 0.382)

    # Look for entry point F (green candle above fibo_CD)
    for i in range(ind_E, len(Data)):
        if Data.loc[i, 'low'] < fail_level: return []
        if Data.loc[i, 'close'] > Data.loc[i, 'open'] and Data.loc[i, 'close'] > fibo_CD:
            return [ind_A, ind_B, ind_C, ind_D, ind_E, i]         
    return []

def strategy_entry(Data):
    """ Executes entry strategy on data, where data is a dataframe with open,high,low,close,volume as columns.
        Returns list of entry points as tuples of indexes [A, B, C, D, E, F]
    """
    Data = add_fractals(Data)
    entries = []
    
    for i in range(len(Data)):
        if Data.loc[i, 'fractal_down']:
            entry_points = find_entry_from(Data, i)
            if len(entry_points) > 0: entries.append(entry_points)

    # Sort entries by index
    entries = list(set(tuple(lst) for lst in entries))
    entries.sort()
    return entries


def strategy_exit(Data, start, entry_points, SL_style='optimal', TP_level=2.618):
    """ Given prices at entry points, executes exit strategy on data. Looking for an exit starting from start index.
        Returns tuple with index of exit and type of exit (TP or SL) if exit was found
    """
    A, B, C, D, E, F = entry_points
    
    if SL_style == 'out':
        SL = fibo_retracement(A, B, 1.272)
    elif SL_style == 'optimal':
        SL = fibo_retracement(A, B, 0.855)
    elif SL_style == 'aggressive':
        SL = fibo_retracement(C, D, 0.855)

    TP = fibo_retracement(D, E, TP_level)
    
    for i in range(start, len(Data)):
        if Data.loc[i, 'high'] > TP:
            return (i, 'TP')
        elif Data.loc[i, 'low'] < SL:
            return (i, 'SL')
    return None


def entries_to_txt(entries, filename):
    """ Writes entries to a text file """
    with open(filename, 'w') as f:
        for entry in entries:
            f.write(str(entry) + '\n')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_csv_file containing price history>")
        sys.exit(1)

    data = pd.read_csv(sys.argv[1])
    entries = strategy_entry(data)
    entries_to_txt(entries, 'entries.txt') # writes entries to txt in current directory 