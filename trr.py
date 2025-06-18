import sys
import numpy as np
from struct import unpack


class TRRHeader:
    """
    Parse the header of a GROMACS .trr trajectory frame from a binary stream.

    Attributes:
        tag (bytes): Magic number and padded string identifier.
        header (list): Parsed integer fields from the header.
        step (int): Simulation step number from the header.
        double (bool): Whether the file uses double precision floats.
        time (float): Simulation time of the frame.
        fsize (int): Size of frame data in bytes (excluding header).
        bsize (int): Size of the largest data block (coordinates, velocities, or forces).
        hsize (int): Size of the header in bytes (120 or 164).
    """

    def __init__(self, stream):
        """
        Initialize and parse a TRRHeader from the binary stream.

        Args:
            stream (file-like): Open binary stream positioned at the start of a .trr frame.
        """
        # Header:
        #  0     0    l - Magic number (1993)
        #  1     4    l - Size of string (13)
        # ...         S - Padded string (16 characters)
        #  6    24  0 l - Size of input record (0)
        #  7    28  1 l - Size of energy record (0)
        #  8    32  2 l - Size of box (36 or 72)
        #  9    36  3 l - Size of virial tensor (0)
        # 10 -9 40  4 l - Size of pressure tensor (0)
        # 11 -8 44  5 l - Size of topology (0)
        # 12 -7 48  6 l - Size of symmetry (0)
        # 13 -6 52  7 l - Size of coordinates X
        # 14 -5 56  8 l - Size of velocities V
        # 15 -4 60  9 l - Size of forces F
        # 16 -3 64 10 l - Number of atoms
        # 17 -2 68 11 l - Step
        # 18 -1 72 12 l - NRE
        #       76
        header = stream.read(76)
        self.tag = header[:8]
        self._rawheader = list(unpack('>lllllllllllll', header[24:]))
        self.step = self._rawheader[-2]

        # Deatermine if real is single or double precision
        # Assume the trajectory will have at least one of X, V, F
        self.double = (max(self._rawheader[-6:-3]) // self._rawheader[-3] // 3) == 8

        # 19    76     f - Time
        # 20    80/84  f - Lambda
        #       84/92

        if self.double:
            self._rawheader.extend(unpack('>dd', stream.read(16)))
        else:
            self._rawheader.extend(unpack('>ff', stream.read(8)))

        self.positions = self._rawheader[7]
        self.velocities = self._rawheader[8]
        self.forces = self._rawheader[9]
        self.natoms = self._rawheader[10]
        self.time = self._rawheader[13]
        self.lambd = self._rawheader[14]
        
        self.fsize = sum(self._rawheader[:10]) # Including box, x, v, f
        self.bsize = max(self._rawheader[7:10])  # Size of block (x, v, or f)
        self.hsize = 164 if self.double else 120 
        

class TRR:
    """
    Read and index GROMACS .trr trajectory frames.

    Attributes:
        trr (file): Open .trr file in binary mode.
        frames (list): Byte positions of each frame in the file.
        headers (list): TRRHeader objects for each frame.
    """

    def __init__(self, filename, selection=slice(None)):
        """
        Initialize a TRR reader from a GROMACS .trr trajectory file, index all frames,
        and determine maximum atom index based on an optional atom selection.

        Args:
            filename (str): Path to the .trr binary trajectory file.
            selection (int, slice, list, tuple, np.ndarray, optional): Atom indices or boolean mask
                specifying which atoms to consider. Used to determine the highest accessed atom index.

        Attributes:
            trr (file): Open binary file handle.
            frames (list of int): Byte offsets of each frame in the file.
            headers (list of TRRHeader): Parsed headers for each frame.
            natoms (int): Total number of atoms in the system.
            selection: The provided atom selection.
            _maxidx (int): Highest atom index accessed, inferred from selection.
        """
        self.filename = filename
        self.trr = open(filename, 'rb')
        
        size = self.trr.seek(0, 2) and self.trr.tell()
        frames = []
        self.headers = []
        
        # Read the positions of the frames and the corresponding headers
        pos = 0                
        while pos < size:
            self.trr.seek(pos, 0)
            frames.append(pos)
            self.headers.append(TRRHeader(self.trr))
            pos += self.headers[-1].fsize + self.headers[-1].hsize - 36 # No box?

        self.frames = np.array(frames)
        self.times = np.array([ h.time for h in self.headers ])
        
        # These are the sizes of each section per frame
        self._sizes = np.array([ 
            (h.positions, h.velocities, h.forces)
            for h in self.headers 
        ]).T
        self._have = self._sizes.astype(bool)
        
        # Determine the positions of each section in each frame (if present)
        self._starts = np.zeros_like(self._have) + frames + self.headers[0].hsize
        self._starts[1:] += self._sizes[0]
        self._starts[2] += self._sizes[1]
                     
        self.natoms = self.headers[0].natoms
        self.selection = selection
        
        # Determine the maximum atom index to read from the selection
        self._maxidx = self.natoms
        if isinstance(selection, int):
            self._maxidx = min(selection, self._maxidx)
            self.selection = slice(0, selection)
        elif isinstance(selection, slice):
            if selection.stop is not None:
                self._maxidx = min(selection.stop, self._maxidx)
        elif isinstance(selection, np.ndarray):
            if selection.dtype == bool:
                self._maxidx = np.where(selection)[0][-1]
            else:
                self._maxidx = max(selection) + 1
        elif isinstance(selection, (list, tuple)):
            selmax = max(selection) + 1
            if selmax > 2:
                self._maxidx = selmax
            else:
                self._maxidx = np.where(selection)[0][-1]
    
    def _read(self, k):
        """
        Read a data block (coordinates, velocities, or forces) for all available frames.

        Args:
            k (int): Index of the block type to read (0 = coordinates, 1 = velocities, 2 = forces).

        Returns:
            np.ndarray: Array of shape (n_frames, n_atoms, 3) containing the requested data.
        """
        X = np.zeros((sum(self._have[k]), self._maxidx, 3))
        for frame, pos in enumerate(self._starts[k, self._have[k]]):
            self.trr.seek(pos)
            X[frame] = np.fromfile(self.trr, dtype='>f4', count=3*self._maxidx).reshape((-1, 3))
        return X[:, self.selection]

    @property
    def positions(self):
        """
        np.ndarray: Lazily loaded array of atomic positions for each frame.

        Returns:
            np.ndarray of shape (n_frames, n_atoms, 3).
        """
        if not hasattr(self, '_positions'):
            self._positions = self._read(0)
        return self._positions

    @property
    def velocities(self):
        """
        np.ndarray: Lazily loaded array of atomic velocities for each frame.

        Returns:
            np.ndarray of shape (n_frames, n_atoms, 3).
        """
        if not hasattr(self, '_velocities'):
            self._velocities = self._read(1)
        return self._velocities

    @property
    def forces(self):
        """
        np.ndarray: Lazily loaded array of atomic forces for each frame.

        Returns:
            np.ndarray of shape (n_frames, n_atoms, 3).
        """
        if not hasattr(self, '_forces'):
            self._forces = self._read(2)
        return self._forces

    
if __name__ == '__main__':
    import sys
    t = TRR(sys.argv[1])


