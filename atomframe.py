"""Functions for reading and writing PDB files, and converting to/from Rosetta Pose.
"""
import gzip
import io
import logging
import os
import re
import string
import tempfile
from collections import defaultdict
from glob import glob

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# constants copied here so this module can be stand-alone

AA_3 = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
           'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
AA_1 = list('ARNDCQEGHILKMFPSTWYV')
CANONICAL_AA = AA_1
AA_3_1 = dict(zip(AA_3, AA_1))
AA_1_3 = dict(zip(AA_1, AA_3))

alphabet = 'abcdefghijklmnopqrstuvwxyz'
digits = '0123456789'
all_chain_ids = alphabet.upper() + alphabet.lower() + digits


def read_pdb(filename, add_info=True, reorder_cols=True, multimodel=False):
    """

    :param filename: path to .pdb or .pdb.gz
    :param add_info: whether to add res_ix, res_aa, res_ix_global
    :param reorder_cols: whether to apply logical column ordering
    :param multimodel: if True and multiple models are present, return a 
        list
    """
    if filename.endswith('gz'):
        txt = gzip.open(filename, 'rt').read()
    else:
        txt = open(filename, 'r').read()
    return read_pdb_string(
        txt, reorder_cols=reorder_cols, add_info=add_info, 
        multimodel=multimodel)


def read_pdb_string(pdb_string, reorder_cols=True, add_info=True, 
        multimodel=False):
    models = pdb_string.split('ENDMDL')
    models = [m for m in models if 'ATOM' in m]
    if len(models) > 1 and not multimodel:
        msg = f'{len(models)} models detected, loading the first one'
        logger.warning(msg)
        models = [models[0]]

    arr = []
    for model in models:
        line_filter = '^ATOM'
        txt = [x for x in model.split('\n') if re.match(line_filter, x)]
        df = read_pdb_records('\n'.join(txt))

        if reorder_cols:
            df = df[pdb_useful_order]

        if add_info:
            df = (df
                .sort_values('atom_serial')
                .assign(res_ix=lambda x: x['res_seq'].astype('category').cat.codes)
                .assign(res_aa=lambda x: x['res_name'].map(AA_3_1))
                .assign(res_ix_global=get_global_residue_index)
                )
        arr += [df]

    if len(arr) == 1:
        return df
    return arr


def get_global_residue_index(df_pdb):
    seen = set()
    i = -1
    arr = []
    for res_seq, chain in df_pdb[['res_seq', 'chain']].values:
        if (res_seq, chain) not in seen:
            i += 1
            seen.add((res_seq, chain))
        arr += [i]
    return arr


def update_res_ix_global(df_pdb):
    return df_pdb.assign(res_ix_global=get_global_residue_index)


def read_pdb_records(pdb_string, only_cols=None):
    """
    http://www.wwpdb.org/
     documentation/file-format-content/format33/sect9.html
    """
    def cast_res_seq(df, col_res_seq='res_seq', col_serial='atom_serial'):
        # 4Q2Z_H has non-integer resSeq entries...
        arr = []
        bullshit = False
        bullshit_count = 0
        for i, x in enumerate(df[col_res_seq]):
            try:
                arr.append(int(x))
            except ValueError:
                if not bullshit:
                    bullshit = x, df.iloc[i][col_serial]
                bullshit_count += 1
                arr.append(np.nan)

        if bullshit:
            resSeq, serial = bullshit
            msg = (f'bullshit detected starting at '
                   f'resSeq={resSeq}, serial={serial}; ' 
                   f'dropped {bullshit_count} records')
            logger.warning(msg)

        return (df.assign(**{col_res_seq: arr})
            .dropna(subset=[col_res_seq])
            .assign(**{col_res_seq: lambda x: x[col_res_seq].astype(int)})
        )

    columns, colspecs = get_pdb_colspecs()

    buffer = io.StringIO(pdb_string)
    return (pd.read_fwf(buffer, colspecs=colspecs, header=None)
            .rename(columns={i: x for i, x in enumerate(columns)})
            .drop('', axis=1)
            .pipe(cast_res_seq)
            .assign(iCode=lambda x: x['iCode'].fillna(''))
            .assign(altLoc=lambda x: x['altLoc'].fillna(''))
            .assign(charge=lambda x: x['altLoc'].fillna(''))
    )


def get_pdb_colspecs():
    """Column names and widths for `pd.read_fwf`.
    """
    col_widths = [x[2] for x in pdb_spec]
    col_cs = np.cumsum(col_widths)
    colspecs = list(zip([0] + list(col_cs), col_cs))
    columns = [x[1] for x in pdb_spec]
    return columns, colspecs


# name in spec, name, width, format
pdb_spec = [
    # this is spec, but some code writes atom_serial over 100,000...
    # ('Record name','record_name',    6, '{ <6}'),
    # ('serial',     'atom_serial',    5, '{ >5}'),
    
    ('Record name','record_name',    5, '{ <5}'),
    ('serial',     'atom_serial',    6, '{ >6}'),

    ('unused',     '',               1, ' '),
    ('name',       'atom_name',      4, '{ <4}'),
    ('altLoc',     'altLoc',         1, '{ <1}'),
    ('resName',    'res_name',       3, '{ >3}'),
    ('unused',     '',               1, ' '),
    ('chainID',    'chain',          1, '{ <1}'),
    ('resSeq',     'res_seq',        4, '{ >4}'),
    ('iCode',      'iCode',          1, '{ <1}'),
    ('unused',     '',               3, '   '),
    ('x',          'x',              8, 'fw6'),
    ('y',          'y',              8, 'fw6'),
    ('z',          'z',              8, 'fw6'),
    ('occupancy',  'occupancy',      6, 'fw4'),
    ('tempFactor', 'temp_factor',    6, 'fw4'),
    ('unused',     '',               6, ' '*6),
    ('segmentID',  '',               4, ' '*4),
    ('element',    'element',        2, '{>2}'),
    ('charge',     'charge',         2, '{>2}'),
    ]



pdb_useful_order = [
   # yes
   'atom_serial', 'atom_name', 'res_name',
   'chain', 'res_seq', 'x', 'y', 'z', 
   # maybe
   'occupancy', 'temp_factor', 'element',
   # no
   'record_name', 'iCode', 'altLoc', 'charge'
   ]


def atom_record(atom_name, atom_serial, 
    res_name, chain, res_seq, x, y, z, 
    element, altLoc=' ', iCode=' ', occupancy=1, temp_factor=0, charge='', 
    **junk):
    record_name = 'ATOM'
    fields = []
    for pdb_name, name, width, fmt in pdb_spec:
        if name == '':
            fields.append(fmt)
        elif fmt.startswith('fw'):
            fmt_width = int(fmt[2])
            pad = ' ' * (width - fmt_width)
            number = to_fixed_width(locals()[name], fmt_width)
            fields.append(pad + number)
        else:
            fmt_py = fmt[0] + name + ':' + fmt[1:]
            fields.append(fmt_py.format(**locals()))
        
    return ''.join(fields)


def write_pdb(df, filename, pipe=True, makedir=True):
    pdbstring = dataframe_to_pdbstring(df)
    ext = os.path.splitext(filename)[1]
    if ext == '':
        filename += '.pdb'
    if makedir:
        dirname = os.path.dirname(os.path.abspath(filename))
        os.makedirs(dirname, exist_ok=True)
    with open(filename, 'w') as fh:
        fh.write(pdbstring)
    if pipe:
        return df


def dataframe_to_pdbstring(df):
    cols = list(df)
    lines = []
    for row in df.values:
        row = dict(zip(cols, row))
        lines.append(atom_record(**row))
    pdbstring = '\n'.join(lines)
    return pdbstring


def dataframe_to_pose(df):
    from pyrosetta import Pose
    from pyrosetta.rosetta.core.import_pose import \
        pose_from_pdbstring  # type: ignore

    serial_ids = df['atom_serial']
    if len(serial_ids) != len(set(serial_ids)):
        raise ValueError('serial IDs not unique')
    pdbstring = dataframe_to_pdbstring(df)
    pose = Pose()
    pose_from_pdbstring(pose, pdbstring)
    return pose


def test_pdb_roundtrip(files, max_numeric_error=0.1):
    """Roundtrip pdb files within numeric error.

    >>>debug code
    ix = 169
    pd.concat([df.iloc[ix], df2.iloc[ix]], axis=1)

    # copied from less
    entry = 'ATOM  25494  OE2 GLU F  74    -108.874   4.135 -58.740  1.00378.47           O'
    record = diy.atom_record(**df.iloc[ix])
    print(entry)
    print(record)

    len(entry), len(record)

    for i, (c1, c2) in enumerate(zip(entry, record)):
        print(i+1, c1, c2)

    """
    test_filename = os.path.join(tempfile.tempdir, 'test.pdb')

    for f in files:
        df = read_pdb(f)
        write_pdb(df, test_filename)
        df2 = read_pdb(test_filename)

        for col in df:
            if np.issubdtype(df[col].dtype, np.number):
                assert ((df[col] - df2[col]).abs() < max_numeric_error).all()
            else:
                assert (df[col] == df2[col]).all()


def pose_to_dataframe(pose):
    """Is it safe to assume the order of residue entries from `to_pdb_string`
    reflects the Rosetta residue numbering?

    "res_ix" is the zero-indexed order of residues, as in `pose.sequence()`
    """
    # ridiculous workaround for pyrosetta.distributed assertion
    import sys
    from collections import namedtuple

    from pyrosetta.distributed.io import to_pdbstring  # type: ignore
    version_info = sys.version_info
    vi = namedtuple('jfc', ['major', 'minor', 'micro', 'releaselevel', 'serial'])
    sys.version_info = vi(major=3, minor=7, micro=7, releaselevel='final', serial=0)
    pdb_string = to_pdbstring(pose)
    assert not isinstance(pdb_string, tuple), 'no pyrosetta overloading'
    df_pdb = read_pdb_string(pdb_string)
    sys.version_info = version_info

    return (df_pdb
        .pipe(validate_pose_dataframe_numbering, pose)
        .assign(res_ix=lambda x: x.groupby(['chain', 'res_seq']).ngroup())
        )


def validate_pose_dataframe_numbering(df_pdb, pose):
    """Make sure the residue number ("res_seq", one-indexed) and chain match.
    """
    pdb_numbering = df_pdb[['chain', 'res_seq']].drop_duplicates().values
    pdb_info = pose.pdb_info()
    for i in range(1, 1 + pose.total_residue()):
        res_seq, chain = pdb_info.pose2pdb(i).split()
        res_seq = int(res_seq)
        assert chain == pdb_numbering[i - 1][0]
        assert res_seq == pdb_numbering[i - 1][1]
    
    return df_pdb


def pdb_frame(files_or_search, col_file='file', progress=None):
    """Convenience function, pass either a list of files or a 
    glob wildcard search term.
    """
    if progress is None:
        progress = lambda x: x
    
    from natsort import natsorted
    if isinstance(files_or_search, str):
        files = natsorted(glob(files_or_search))
    else:
        files = files_or_search

    return pd.concat([read_pdb(f).assign(**{col_file: f}) 
        for f in progress(files)], sort=False)


def debug_parsing(entry, record):
    print(entry)
    print(record)

    for i, (c1, c2) in enumerate(zip(entry, record)):
        print(f'{i+1: >2} {c1} {c2}')


def debug_pdb_spec(pdb_spec):
    i = 1
    for name, _, width, _ in pdb_spec:
        print(i, '-', i + width - 1, name)
        i = i + width


def first_altloc(df_pdb):
    """Boolean mask corresponding to first listed altLoc for each residue.
    """
    it = df_pdb[['res_seq', 'altLoc', 'atom_name']].values
    current_seq = None
    current_altLoc = None
    mask = []
    for res_seq, altLoc, atom_name in it:
        if res_seq != current_seq:
            current_seq = res_seq
            current_altLoc = altLoc
        if altLoc != current_altLoc:
            mask.append(False)
        else:
            mask.append(True)
    return mask


def read_rosetta_params(filename):
    
    tables = {'ICOOR_INTERNAL': 
              ('child', 'phi', 'theta', 'distance', 'parent', 
              'angle', 'torsion'),
              'ATOM':
              ('res_name_alias', 'Rosetta atom type', 
               'CHARMM atom type', 'partial_charge', 
               'partial charge 2 (?)'),
              'CHI': ('chi_num', 'a', 'b', 'c', 'd'),
              'ATOM_ALIAS': ('atom_name_alias', 'atom_name'),
             }
    with open(filename, 'r') as fh:
        lines = fh.readlines()

    for line in lines:
        if line.startswith('NAME '):
            name = line.split()[1]
    
    results = dict(res_name=name)
    for table, columns in tables.items():
        body = '\n'.join(x for x in lines if x.startswith(table))
        if body:
            df = pd.read_csv(io.StringIO(body), sep='\s+', 
                             header=None)
            df = df.iloc[:, 1:]
            df.columns = columns[:df.shape[1]]
            results[table] = df
        
    return results


def to_fixed_width(n, max_width, allow_overflow=False, do_round=True):
    """https://stackoverflow.com/questions/24960235/python-how-do-i-format-numbers-for-a-fixed-width
    """
    if do_round:
        for i in range(max_width - 2, -1, -1):
            str0 = '{:.{}f}'.format(n, i)
            if len(str0) <= max_width:
                break
    else:
        str0 = '{:.42f}'.format(n)
        int_part_len = str0.index('.')
        if int_part_len <= max_width - 2:
            str0 = str0[:max_width]
        else:
            str0 = str0[:int_part_len]
    if (not allow_overflow) and (len(str0) > max_width):
        raise OverflowError(
            "Impossible to represent in fixed-width non-scientific format")
    return str0


def get_chain_sequences(df_pdb):
    return (df_pdb.drop_duplicates(['res_seq', 'chain'])
            .groupby('chain')['res_aa'].apply(''.join).to_dict())


def parse_scorefile_string(txt):
    txt = re.sub('^SCORE:\s+', '', txt, flags=re.MULTILINE)
    return pd.read_csv(io.StringIO(txt), sep='\s+')


def read_scorefile(f):
    """Rosetta .sc files
    """
    with open(f, 'r') as fh:
        return parse_scorefile_string(fh.read())


def read_pdb_sequences(filename, first_chain_only=False):
    """Faster loading of chain residue sequences from pdb.
    """
    if filename.endswith('gz'):
        fh = gzip.open(filename, 'rt')
    else:
        fh = open(filename, 'r')
    return parse_pdb_sequences(fh, first_chain_only=first_chain_only)


def parse_pdb_sequences(filehandle, first_chain_only=False):
    """Quickly read chain sequences from an iterator of lines.
    Assumes that residue number increases within each chain (doesn't
    have to start from 1 or be sequential).
    """
    columns = dict(zip(*get_pdb_colspecs()))
    chain_0, chain_1 = columns['chain']
    res_0, res_1 = columns['res_name']
    res_seq_0, res_seq_1 = columns['res_seq']
    chains = defaultdict(list)
    chain_lengths = defaultdict(lambda: -1)
    for line in filehandle:
        if line.startswith('ATOM'):
            chain = line[chain_0:chain_1]
            aa = AA_3_1.get(line[res_0:res_1], 'x')
            res_seq = line[res_seq_0:res_seq_1]
            res_i = int(res_seq) - 1
            if res_i > chain_lengths[chain]:
                chain_lengths[chain] = res_i
                chains[chain].append(aa)
        if first_chain_only and line.startswith('TER'):
            break
    chains = {k: ''.join(v) for k,v in chains.items()}
    
    return chains
        

def fix_weird_residues(df_pdb):
    """Correct weird residue names 
    """
    weirdos = {'KRR': 'K'}
    arr = []
    for a, b in df_pdb[['res_name', 'res_aa']].values:
        if a in weirdos:
            arr.append(weirdos[a])
        else:
            arr.append(b)
    return (df_pdb.assign(res_aa=arr))


def renumber_residues(pose):
    pdbi = pose.pdb_info()
    for i in range(pose.size()):
        pdbi.number(i+1, i+1)


def renumber_residues_atomframe(af, include_atoms=False):
    af['res_ix'] = af.groupby(['chain', 'res_seq']).ngroup()
    af['res_ix_global'] = af['res_ix']
    af['res_seq'] = af['res_ix'] + 1
    if include_atoms:
        af['atom_serial'] = np.arange(af.shape[0]) + 1
    return af


def hash_atomframe(df_atoms):
    xyz = hash(tuple(df_atoms[['x', 'y', 'z']].round(3).values.flat))
    res = hash(df_atoms['res_name'].sum())

    return alpha_hash(xyz + res, HASH_WIDTH)


def transform_atomframe(df_atoms, xform):
    coords = df_atoms.assign(w=1)[['x', 'y', 'z', 'w']].values
    coords_ = coords @ xform.T
    return df_atoms.assign(x=coords_[:, 0], y=coords_[:, 1], z=coords_[:, 2])


def concatenate_chains(*atomframes):
    """Reassigns chains starting from A. Updates atom_serial but does
    not change residue numbering.
    """
    atomframes = [x[1] for df in atomframes for x in df.groupby('chain')]
    it = zip(atomframes, string.ascii_uppercase)
    return (pd.concat([x.assign(chain=c) for x,c in it])
     .assign(atom_serial=lambda x: 1 + np.arange(x.shape[0]))
     .pipe(update_res_ix_global)
     )


def fix_numbering(df_atoms):
    """Reassigns atom_serial and closes gaps in monotonically 
    increasing res_seq.
    """
    return (df_atoms
     .sort_values(['chain', 'atom_serial', 'res_seq'])
     .assign(atom_serial=lambda x: 1 + np.arange(x.shape[0]))
     .assign(res_seq=lambda x: 1 + np.unique(x['res_seq'], return_inverse=True)[1])
    )


def write_multimodel_pdb(filename, *atomframes, 
        reassign_chains=True, makedir=True):
    ext = os.path.splitext(filename)[1]
    if ext == '':
        filename += '.mm.pdb'
    if makedir:
        dirname = os.path.dirname(os.path.abspath(filename))
        os.makedirs(dirname, exist_ok=True)
    # reassign chains
    alphabet = iter(string.ascii_uppercase)
    txt = []
    for i, af in enumerate(atomframes):
        txt += [f'MODEL        {i + 1}']
        if reassign_chains:
            chain_map = {k: next(alphabet) for k in af['chain'].drop_duplicates()}
            txt.append(af
             .assign(chain=lambda x: x['chain'].map(chain_map))
             .pipe(dataframe_to_pdbstring)
            )
        else:
            txt += [dataframe_to_pdbstring(af)]
        txt += ['ENDMDL']
        
    with open(filename, 'w') as fh:
        fh.write('\n'.join(txt))


def calculate_residue_distances(atomframe):
    atomframe = update_res_ix_global(atomframe)
    atom_counts = (atomframe
     .drop_duplicates(['res_ix_global', 'atom_name'])
     ['atom_name'].value_counts())
    check_counts = atom_counts['N'] == atom_counts['C'] == atomframe['res_ix_global'].max() + 1
    assert check_counts, 'all residues must have N and C defined'

    xyz = (atomframe
     .query('atom_name == ["C", "N"]')
     .sort_values(['res_ix_global', 'atom_name'], ascending=(True, False))
     [['x', 'y', 'z']]
    )
    distances = (xyz.diff()**2).sum(axis=1)**0.5
    return distances[2::2]
