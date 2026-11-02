"""
Module to read MODFLOW binary output files.  The module contains four
important classes that can be accessed by the user.

*  HeadFile (Binary head file.  Can also be used for drawdown)
*  HeadUFile (Binary MODFLOW-USG unstructured head file)
*  UcnFile (Binary concentration file from MT3DMS)
*  CellBudgetFile (Binary cell-by-cell flow file)

"""
import os
from pathlib import Path
from typing import Optional, Union
import warnings

import numpy as np

from mfobs.datafile import Header, LayerFile

warnings.simplefilter("always", DeprecationWarning)


class FlopyBinaryData:
    """
    The FlopyBinaryData class is a class to that defines the data types for
    integer, floating point, and character data in MODFLOW binary
    files. The FlopyBinaryData class is the super class from which the
    specific derived classes are formed.  This class should not be
    instantiated directly.

    """

    def __init__(self):
        self.integer = np.int32
        self.integerbyte = self.integer(1).nbytes

        self.character = np.uint8
        self.textbyte = 1

        return

    def set_float(self, precision):
        self.precision = precision
        if precision.lower() == "double":
            self.real = np.float64
            self.floattype = "f8"
        else:
            self.real = np.float32
            self.floattype = "f4"
        self.realbyte = self.real(1).nbytes
        return

    def read_text(self, nchar=20):
        bytesvalue = self._read_values(self.character, nchar).tobytes()
        return bytesvalue.decode().strip()

    def read_integer(self):
        return self._read_values(self.integer, 1)[0]

    def read_real(self):
        return self._read_values(self.real, 1)[0]

    def read_record(self, count, dtype=None):
        if dtype is None:
            dtype = self.dtype
        return self._read_values(dtype, count)

    def _read_values(self, dtype, count):
        return np.fromfile(self.file, dtype, count)
    
    
class MfGrdFile(FlopyBinaryData):
    """
    The MfGrdFile class.

    Parameters
    ----------
    filename : str
        Name of the MODFLOW 6 binary grid file
    precision : string
        'single' or 'double'.  Default is 'double'.
    verbose : bool
        Write information to standard output.  Default is False.

    Attributes
    ----------

    Methods
    -------

    See Also
    --------

    Notes
    -----
    The MfGrdFile class provides simple ways to retrieve data from binary
    MODFLOW 6 binary grid files (.grb). The binary grid file contains data
    that can be used for post processing MODFLOW 6 model results. For
    example, the ia and ja arrays for a model grid.

    Examples
    --------
    >>> import flopy
    >>> gobj = flopy.utils.MfGrdFile('test.dis.grb')
    """

    def __init__(self, filename, precision="double", verbose=False):
        """
        Class constructor.

        """

        # Call base class init
        super().__init__()

        # set attributes
        self.precision = precision
        self.verbose = verbose
        self._initial_len = 50
        self._recorddict = {}
        self._datadict = {}
        self._recordkeys = []
        self.filename = filename

        if self.verbose:
            print(f"\nProcessing binary grid file: {filename}")

        # open the grb file
        self.file = open(filename, "rb")

        # grid type
        line = self.read_text(self._initial_len).strip()
        t = line.split()
        self._grid_type = t[1]

        # version
        line = self.read_text(self._initial_len).strip()
        t = line.split()
        self._version = t[1]

        # version
        line = self.read_text(self._initial_len).strip()
        t = line.split()
        self._ntxt = int(t[1])

        # length of text
        line = self.read_text(self._initial_len).strip()
        t = line.split()
        self._lentxt = int(t[1])

        # read text strings
        for idx in range(self._ntxt):
            line = self.read_text(self._lentxt).strip()
            t = line.split()
            key = t[0]
            dt = t[1]
            if dt == "INTEGER":
                dtype = np.int32
            elif dt == "SINGLE":
                dtype = np.float32
            elif dt == "DOUBLE":
                dtype = np.float64
            else:
                dtype = None
            nd = int(t[3])
            if nd > 0:
                shp = [int(v) for v in t[4:]]
                shp = tuple(shp[::-1])
            else:
                shp = (0,)
            self._recorddict[key] = (dtype, nd, shp)
            self._recordkeys.append(key)
            if self.verbose:
                s = ""
                if nd > 0:
                    s = shp
                print(f"  File contains data for {key} with shape {s}")

        if self.verbose:
            print(f"Attempting to read {self._ntxt} records from {filename}")

        for key in self._recordkeys:
            if self.verbose:
                print(f"  Reading {key}")
            dt, nd, shp = self._recorddict[key]
            # read array data
            if nd > 0:
                count = 1
                for v in shp:
                    count *= v
                v = self.read_record(count=count, dtype=dt)
            # read variable data
            else:
                if dt == np.int32:
                    v = self.read_integer()
                elif dt == np.float32:
                    v = self.read_real()
                elif dt == np.float64:
                    v = self.read_real()
            self._datadict[key] = v

            if self.verbose:
                if nd == 0:
                    print(f"  {key} = {v}")
                else:
                    print(f"  {key}: min = {v.min()} max = {v.max()}")

        # close the file
        self.file.close()

        # initialize the model grid to None
        self.__modelgrid = None

        # set ia and ja
        self.__set_iaja()

    # internal functions
    def __set_iaja(self):
        """
        Set ia and ja from _datadict.
        """
        self._ia = self._datadict["IA"] - 1
        self._ja = self._datadict["JA"] - 1

    def __set_modelgrid(self):
        """
        Define structured, vertex, or unstructured grid based on MODFLOW 6
        discretization type.

        Returns
        -------
        modelgrid : grid
        """
        from ...discretization.structuredgrid import StructuredGrid
        from ...discretization.unstructuredgrid import UnstructuredGrid
        from ...discretization.vertexgrid import VertexGrid

        modelgrid = None
        idomain = self.idomain
        xorigin = self.xorigin
        yorigin = self.yorigin
        angrot = self.angrot

        try:
            top = self.top
            botm = self.bot

            if self._grid_type == "DISV":
                nlay, ncpl = self.nlay, self.ncpl
                vertices, cell2d = self.cell2d
                top = np.ravel(top)
                botm.shape = (nlay, ncpl)
                modelgrid = VertexGrid(
                    vertices,
                    cell2d,
                    top,
                    botm,
                    idomain,
                    xoff=xorigin,
                    yoff=yorigin,
                    angrot=angrot,
                )

            elif self._grid_type == "DIS":
                nlay, nrow, ncol = (
                    self.nlay,
                    self.nrow,
                    self.ncol,
                )
                delr, delc = self.delr, self.delc

                top.shape = (nrow, ncol)
                botm.shape = (nlay, nrow, ncol)
                modelgrid = StructuredGrid(
                    delc,
                    delr,
                    top,
                    botm,
                    idomain=idomain,
                    xoff=xorigin,
                    yoff=yorigin,
                    angrot=angrot,
                )
            else:
                iverts, verts = self.iverts, self.verts
                vertc = self.cellcenters
                xc, yc = vertc[:, 0], vertc[:, 1]
                modelgrid = UnstructuredGrid(
                    vertices=verts,
                    iverts=iverts,
                    xcenters=xc,
                    ycenters=yc,
                    top=top,
                    botm=botm,
                    idomain=idomain,
                    xoff=xorigin,
                    yoff=yorigin,
                    angrot=angrot,
                )

        except:
            print(f"could not set model grid for {self.file.name}")

        self.__modelgrid = modelgrid

        return

    def __build_vertices_cell2d(self):
        """
        Build the mf6 vertices and cell2d array to generate a VertexGrid

        Returns:
        -------
            vertices: list
            cell2d: list
        """
        iverts, verts = self.iverts, self.verts
        vertc = self.cellcenters
        vertices = [[ix] + list(i) for ix, i in enumerate(verts)]
        cell2d = [
            [ix] + list(vertc[ix]) + [len(i) - 1] + i[:-1]
            for ix, i in enumerate(iverts)
        ]
        return vertices, cell2d

    def __get_iverts(self):
        """
        Get a list of the vertices that define each model cell.

        Returns
        -------
        iverts : list of lists
            List with lists containing the vertex indices for each model cell.

        """
        iverts = None
        if "IAVERT" in self._datadict:
            if self._grid_type == "DISV":
                nsize = self.ncpl
            elif self._grid_type == "DISU":
                nsize = self.nodes
            iverts = []
            iavert = self.iavert
            javert = self.javert
            for ivert in range(nsize):
                i0 = iavert[ivert]
                i1 = iavert[ivert + 1]
                iverts.append((javert[i0:i1]).tolist())
            if self.verbose:
                print(f"returning iverts from {self.file.name}")
        return iverts

    def __get_verts(self):
        """
        Get a list of the x, y pair for each vertex from the data in the
        binary grid file.

        Returns
        -------
        verts : np.ndarray
            Array with x, y pairs for every vertex used to define the model.

        """
        verts = None
        if "VERTICES" in self._datadict:
            shpvert = self._recorddict["VERTICES"][2]
            verts = self._datadict["VERTICES"].reshape(shpvert)
            if self._grid_type == "DISU":
                # modify verts
                verts = [
                    [idx, verts[idx, 0], verts[idx, 1]]
                    for idx in range(shpvert[0])
                ]
            if self.verbose:
                print(f"returning verts from {self.file.name}")
        return verts

    def __get_cellcenters(self):
        """
        Get the cell centers centroids for a MODFLOW 6 GWF model that uses
        the DISV or DISU discretization.

        Returns
        -------
        vertc : np.ndarray
            Array with x, y pairs of the centroid for every model cell

        """
        xycellcenters = None
        if "CELLX" in self._datadict:
            x = self._datadict["CELLX"]
            y = self._datadict["CELLY"]
            xycellcenters = np.column_stack((x, y))
            if self.verbose:
                print(f"returning cell centers from {self.file.name}")
        return xycellcenters

    # properties
    @property
    def grid_type(self):
        """
        Grid type defined in the MODFLOW 6 grid file.

        Returns
        -------
        grid_type : str
        """
        return self._grid_type

    @property
    def nlay(self):
        """
        Number of layers. None for DISU grids.

        Returns
        -------
        nlay : int
        """
        if self._grid_type in ("DIS", "DISV"):
            nlay = self._datadict["NLAY"]
        else:
            nlay = None
        return nlay

    @property
    def nrow(self):
        """
        Number of rows. None for DISV and DISU grids.

        Returns
        -------
        nrow : int
        """
        if self._grid_type == "DIS":
            nrow = self._datadict["NROW"]
        else:
            nrow = None
        return nrow

    @property
    def ncol(self):
        """
        Number of columns. None for DISV and DISU grids.

        Returns
        -------
        ncol : int
        """
        if self._grid_type == "DIS":
            ncol = self._datadict["NCOL"]
        else:
            ncol = None
        return ncol

    @property
    def ncpl(self):
        """
        Number of cells per layer. None for DISU grids.

        Returns
        -------
        ncpl : int
        """
        if self._grid_type == "DISV":
            ncpl = self._datadict["NCPL"]
        if self._grid_type == "DIS":
            ncpl = self.nrow * self.ncol
        else:
            None
        return ncpl

    @property
    def ncells(self):
        """
        Number of cells.

        Returns
        -------
        ncells : int
        """
        if self._grid_type in ("DIS", "DISV"):
            ncells = self._datadict["NCELLS"]
        else:
            ncells = self._datadict["NODES"]
        return ncells

    @property
    def nodes(self):
        """
        Number of nodes.

        Returns
        -------
        nodes : int
        """
        if self._grid_type in ("DIS", "DISV"):
            nodes = self.ncells
        else:
            nodes = self._datadict["NODES"]
        return nodes

    @property
    def shape(self):
        """
        Shape of the model grid (tuple).

        Returns
        -------
        shape : tuple
        """
        if self._grid_type == "DIS":
            shape = (self.nlay, self.nrow, self.ncol)
        elif self._grid_type == "DISV":
            shape = (self.nlay, self.ncpl)
        else:
            shape = (self.nodes,)
        return shape

    @property
    def xorigin(self):
        """
        x-origin of the model grid. None if not defined in the
        MODFLOW 6 grid file.

        Returns
        -------
        xorigin : float
        """
        if "XORIGIN" in self._datadict:
            xorigin = self._datadict["XORIGIN"]
        else:
            xorigin = None
        return xorigin

    @property
    def yorigin(self):
        """
        y-origin of the model grid. None if not defined in the
        MODFLOW 6 grid file.

        Returns
        -------
        yorigin : float
        """
        if "YORIGIN" in self._datadict:
            yorigin = self._datadict["YORIGIN"]
        else:
            yorigin = None
        return yorigin

    @property
    def angrot(self):
        """
        Model grid rotation angle. None if not defined in the
        MODFLOW 6 grid file.

        Returns
        -------
        angrot : float
        """
        if "ANGROT" in self._datadict:
            angrot = self._datadict["ANGROT"]
        else:
            angrot = None
        return angrot

    @property
    def idomain(self):
        """
        IDOMAIN for the model grid. None if not defined in the
        MODFLOW 6 grid file.

        Returns
        -------
        idomain : ndarray of ints
        """
        if "IDOMAIN" in self._datadict:
            idomain = self._datadict["IDOMAIN"]
        else:
            idomain = None
        return idomain

    @property
    def delr(self):
        """
        Cell size in the row direction (y-direction). None if not
        defined in the MODFLOW 6 grid file.

        Returns
        -------
        delr : ndarray of floats
        """
        if self.grid_type == "DIS":
            delr = self._datadict["DELR"]
        else:
            delr = None
        return delr

    @property
    def delc(self):
        """
        Cell size in the column direction (x-direction). None if not
        defined in the MODFLOW 6 grid file.

        Returns
        -------
        delc : ndarray of floats
        """
        if self.grid_type == "DIS":
            delc = self._datadict["DELC"]
        else:
            delc = None
        return delc

    @property
    def top(self):
        """
        Top of the model cells in the upper model layer for DIS and
        DISV grids. Top of the model cells for DISU grids.

        Returns
        -------
        top : ndarray of floats
        """
        return self._datadict["TOP"]

    @property
    def bot(self):
        """
        Bottom of the model cells.

        Returns
        -------
        bot : ndarray of floats
        """
        if self.grid_type in ("DIS", "DISV"):
            bot = self._datadict["BOTM"]
        else:
            bot = self._datadict["BOT"]
        return bot

    @property
    def nja(self):
        """
        Number of non-zero entries JA vector array.

        Returns
        -------
        nja : int
        """
        return self._datadict["NJA"]

    @property
    def ia(self):
        """
        index array that defines indexes for `.ja`. Each ia value is the
        starting position of data for a cell. [ia[n]:ia[n+1]] would give you
        all data for a cell. ia[n] is also the location of data for the
        diagonal position. See `.ja` property documentation
        for an example of getting a cell's number and connected cells

        Returns
        -------
        ia : ndarray of ints
        """
        return np.array(self._ia, dtype=int)

    @property
    def ja(self):
        """
        Flat jagged connection array for a model. `.ja` for a cell includes the
        cell number and the cell number for all connected cells. Indexes for
        cells are stored in the `.ia` variable.

        Returns
        -------
        ja : ndarray of ints

        Examples
        --------
        >>> from flopy.mf6.utils import MfGrdFile
        >>> grb = MfGrdFile("my_model.dis.grb")
        >>> ia = grb.ia
        >>> ja = grb.ja
        >>> # get connections for node 0
        >>> ja_node0 = ja[ia[0]:ia[1]]
        >>> node = ja_node0[0]
        >>> connections = ja_node0[1:]
        """
        return self._ja

    @property
    def iavert(self):
        """
        index array that defines indexes for `.javart`. Each ia value is the
        starting position of data for a cell. [iavert[n]:iavert[n+1]] would
        give you all data for a cell. See `.javert` property documentation for
        an example of getting cell number and it's vertex numbers.
        Alternatively, the `.iverts` property can be used to get this
        information

        Returns
        -------
        iavert : ndarray of ints or None for structured grids
        """
        if "IAVERT" in self._datadict:
            iavert = self._datadict["IAVERT"] - 1
        else:
            iavert = None
        return iavert

    @property
    def javert(self):
        """
        Flat jagged array of vertex numbers that comprise all of the cells

        Returns
        -------
        javerts : ndarray of ints or None for structured grids

        Examples
        --------
        >>> from flopy.mf6.utils import MfGrdFile
        >>> grb = MfGrdFile("my_model.dis.grb")
        >>> iavert = self.iavert
        >>> javert = self.javert
        >>> # get vertex numbers for node 0
        >>> vertnums = javert[iavert[0]:iavert[1]]
        """
        if "JAVERT" in self._datadict:
            javert = self._datadict["JAVERT"] - 1
        else:
            javert = None
        return javert

    @property
    def iverts(self):
        """
        Vertex numbers comprising each cell for every cell in model grid.

        Returns
        -------
        iverts : list of lists of ints
        """
        return self.__get_iverts()

    @property
    def verts(self):
        """
        x,y location of each vertex that defines the model grid.

        Returns
        -------
        verts : ndarray of floats
        """
        return self.__get_verts()

    @property
    def cellcenters(self):
        """
        Cell centers (x,y).

        Returns
        -------
        cellcenters : ndarray of floats
        """
        return self.__get_cellcenters()

    @property
    def modelgrid(self):
        """
        Model grid object.

        Returns
        -------
        modelgrid : StructuredGrid, VertexGrid, UnstructuredGrid
        """
        if self.__modelgrid is None:
            self.__set_modelgrid()
        return self.__modelgrid

    @property
    def cell2d(self):
        """
        cell2d data for a DISV grid. None for DIS and DISU grids.

        Returns
        -------
        cell2d : list of lists
        """
        if self._grid_type == "DISV":
            vertices, cell2d = self.__build_vertices_cell2d()
        else:
            vertices, cell2d = None, None
        return vertices, cell2d
    

class BinaryHeader(Header):
    """
    The binary_header class is a class to create headers for MODFLOW
    binary files.

    Parameters
    ----------
        bintype : str
            is the type of file being opened (head and ucn file currently
            supported)
        precision : str
            is the precision of the floating point data in the file

    """

    def __init__(self, bintype=None, precision="single"):
        super().__init__(bintype, precision)

    def set_values(self, **kwargs):
        """
        Set values using kwargs
        """
        ikey = [
            "ntrans",
            "kstp",
            "kper",
            "ncol",
            "nrow",
            "ilay",
            "ncpl",
            "nodes",
            "m2",
            "m3",
        ]
        fkey = ["pertim", "totim"]
        ckey = ["text"]
        for k in ikey:
            if k in kwargs.keys():
                try:
                    self.header[0][k] = int(kwargs[k])
                except:
                    print(
                        f"{k} key not available in {self.header_type} "
                        "header dtype"
                    )
        for k in fkey:
            if k in kwargs.keys():
                try:
                    self.header[0][k] = float(kwargs[k])
                except:
                    print(
                        f"{k} key not available "
                        f"in {self.header_type} header dtype"
                    )
        for k in ckey:
            if k in kwargs.keys():
                # Convert to upper case to be consistent case used by MODFLOW
                # text strings. Necessary to work with HeadFile and UcnFile
                # routines
                ttext = kwargs[k].upper()
                # trim a long string
                if len(ttext) > 16:
                    text = ttext[0:16]
                # pad a short string
                elif len(ttext) < 16:
                    text = "{:<16}".format(ttext)
                # the string is just right
                else:
                    text = ttext
                self.header[0][k] = text
            else:
                self.header[0][k] = "DUMMY TEXT"

    @staticmethod
    def set_dtype(bintype=None, precision="single"):
        """
        Set the dtype

        """
        header = Header(filetype=bintype, precision=precision)
        return header.dtype

    @staticmethod
    def create(bintype=None, precision="single", **kwargs):
        """
        Create a binary header

        """
        header = BinaryHeader(bintype=bintype, precision=precision)
        if header.get_dtype() is not None:
            header.set_values(**kwargs)
        return header.get_values()


def binaryread_struct(file, vartype, shape=(1,), charlen=16):
    """
    Read text, a scalar value, or an array of values from a binary file.

        file : file object
            is an open file object
        vartype : type
            is the return variable type: str, numpy.int32, numpy.float32,
            or numpy.float64
        shape : tuple
            is the shape of the returned array (shape(1, ) returns a single
            value) for example, shape = (nlay, nrow, ncol)
        charlen : int
            is the length of the text string.  Note that string arrays
            cannot be returned, only multi-character strings.  Shape has no
            affect on strings.

    """
    import struct

    import numpy as np

    # store the mapping from type to struct format (fmt)
    typefmtd = {np.int32: "i", np.float32: "f", np.float64: "d"}

    # read a string variable of length charlen
    if vartype == str:
        result = file.read(charlen * 1)

    # read other variable types
    else:
        fmt = typefmtd[vartype]
        # find the number of bytes for one value
        numbytes = vartype(1).nbytes
        # find the number of values
        nval = np.core.fromnumeric.prod(shape)
        fmt = str(nval) + fmt
        s = file.read(numbytes * nval)
        result = struct.unpack(fmt, s)
        if nval == 1:
            result = vartype(result[0])
        else:
            result = np.array(result, dtype=vartype)
            result = np.reshape(result, shape)
    return result


def binaryread(file, vartype, shape=(1,), charlen=16):
    """
    Uses numpy to read from binary file.  This was found to be faster than the
        struct approach and is used as the default.

    """

    # read a string variable of length charlen
    if vartype == str:
        result = file.read(charlen * 1)
    else:
        # find the number of values
        nval = np.prod(shape)
        result = np.fromfile(file, vartype, nval)
        if nval == 1:
            result = result  # [0]
        else:
            result = np.reshape(result, shape)
    return result


def join_struct_arrays(arrays):
    """
    Simple function that can join two numpy structured arrays.

    """
    newdtype = sum((a.dtype.descr for a in arrays), [])
    newrecarray = np.empty(len(arrays[0]), dtype=newdtype)
    for a in arrays:
        for name in a.dtype.names:
            newrecarray[name] = a[name]
    return newrecarray


def get_headfile_precision(filename):
    """
    Determine precision of a MODFLOW head file.

    Parameters
    ----------
    filename : str
    Name of binary MODFLOW file to determine precision.

    Returns
    -------
    result : str
    Result will be unknown, single, or double

    """

    # Set default result if neither single or double works
    result = "unknown"

    # Create string containing set of ascii characters
    asciiset = " "
    for i in range(33, 127):
        asciiset += chr(i)

    # Open file, and check filesize to ensure this is not an empty file
    f = open(filename, "rb")
    f.seek(0, 2)
    totalbytes = f.tell()
    f.seek(0, 0)  # reset to beginning
    assert f.tell() == 0
    if totalbytes == 0:
        raise ValueError(f"datafile error: file is empty: {filename}")

    # first try single
    vartype = [
        ("kstp", "<i4"),
        ("kper", "<i4"),
        ("pertim", "<f4"),
        ("totim", "<f4"),
        ("text", "S16"),
    ]
    hdr = binaryread(f, vartype)
    text = hdr[0][4]
    try:
        text = text.decode()
        for t in text:
            if t.upper() not in asciiset:
                raise Exception()
        result = "single"
        success = True
    except:
        success = False

    # next try double
    if not success:
        f.seek(0)
        vartype = [
            ("kstp", "<i4"),
            ("kper", "<i4"),
            ("pertim", "<f8"),
            ("totim", "<f8"),
            ("text", "S16"),
        ]
        hdr = binaryread(f, vartype)
        text = hdr[0][4]
        try:
            text = text.decode()
            for t in text:
                if t.upper() not in asciiset:
                    raise Exception()
            result = "double"
        except:
            f.close()
            raise ValueError(
                f"Could not determine the precision of the headfile {filename}"
            )

    # close and return result
    f.close()
    return result


class BinaryLayerFile(LayerFile):
    """
    The BinaryLayerFile class is the super class from which specific derived
    classes are formed.  This class should not be instantiated directly

    """

    def __init__(self, filename, precision, verbose, kwargs):
        super().__init__(filename, precision, verbose, kwargs)
        return

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _build_index(self):
        """
        Build the recordarray and iposarray, which maps the header information
        to the position in the binary file.

        """
        header = self._get_header()
        self.nrow = header["nrow"]
        self.ncol = header["ncol"]
        if header["ilay"] > self.nlay:
            self.nlay = header["ilay"]

        if self.nrow < 0 or self.ncol < 0:
            raise Exception("negative nrow, ncol")
        if self.nrow > 1 and self.nrow * self.ncol > 10000000:
            s = "Possible error. ncol ({}) * nrow ({}) > 10,000,000 "
            s = s.format(self.ncol, self.nrow)
            warnings.warn(s)
        self.file.seek(0, 2)
        self.totalbytes = self.file.tell()
        self.file.seek(0, 0)
        ipos = 0
        while ipos < self.totalbytes:
            header = self._get_header()
            self.recordarray.append(header)
            if self.text.upper() not in header["text"]:
                continue
            if ipos == 0:
                self.times.append(header["totim"])
                kstpkper = (header["kstp"], header["kper"])
                self.kstpkper.append(kstpkper)
            else:
                totim = header["totim"]
                if totim != self.times[-1]:
                    self.times.append(totim)
                    kstpkper = (header["kstp"], header["kper"])
                    self.kstpkper.append(kstpkper)
            ipos = self.file.tell()
            self.iposarray.append(ipos)
            databytes = self.get_databytes(header)
            self.file.seek(databytes, 1)
            ipos = self.file.tell()

        # self.recordarray contains a recordarray of all the headers.
        self.recordarray = np.array(self.recordarray, dtype=self.header_dtype)
        self.iposarray = np.array(self.iposarray)
        self.nlay = np.max(self.recordarray["ilay"])
        return

    def get_databytes(self, header):
        """

        Parameters
        ----------
        header : datafile.Header
            header object

        Returns
        -------
         databytes : int
            size of the data array, in bytes, following the header

        """
        return (
            np.int64(header["ncol"])
            * np.int64(header["nrow"])
            * np.int64(self.realtype(1).nbytes)
        )

    def _read_data(self, shp):
        return binaryread(self.file, self.realtype, shape=shp)

    def _get_header(self):
        """
        Read the file header

        """
        header = binaryread(self.file, self.header_dtype, (1,))
        return header[0]

    def get_ts(self, idx):
        """
        Get a time series from the binary file.

        Parameters
        ----------
        idx : tuple of ints, or a list of a tuple of ints
            idx can be (layer, row, column) or it can be a list in the form
            [(layer, row, column), (layer, row, column), ...].  The layer,
            row, and column values must be zero based.

        Returns
        ----------
        out : numpy array
            Array has size (ntimes, ncells + 1).  The first column in the
            data array will contain time (totim).

        See Also
        --------

        Notes
        -----

        The layer, row, and column values must be zero-based, and must be
        within the following ranges: 0 <= k < nlay; 0 <= i < nrow; 0 <= j < ncol

        Examples
        --------

        """
        kijlist = self._build_kijlist(idx)
        nstation = self._get_nstation(idx, kijlist)

        # Initialize result array and put times in first column
        result = self._init_result(nstation)

        istat = 1
        for k, i, j in kijlist:
            ioffset = (i * self.ncol + j) * self.realtype(1).nbytes
            for irec, header in enumerate(self.recordarray):
                ilay = (
                    header["ilay"] - 1
                )  # change ilay from header to zero-based
                if ilay != k:
                    continue
                ipos = int(self.iposarray[irec])

                # Calculate offset necessary to reach intended cell
                self.file.seek(ipos + int(ioffset), 0)

                # Find the time index and then put value into result in the
                # correct location.
                itim = np.where(result[:, 0] == header["totim"])[0]
                result[itim, istat] = binaryread(self.file, self.realtype)
            istat += 1
        return result


class HeadFile(BinaryLayerFile):
    """
    HeadFile Class.

    Parameters
    ----------
    filename : string
        Name of the concentration file
    text : string
        Name of the text string in the head file.  Default is 'head'
    precision : string
        'auto', 'single' or 'double'.  Default is 'auto'.
    verbose : bool
        Write information to the screen.  Default is False.

    Attributes
    ----------

    Methods
    -------

    See Also
    --------

    Notes
    -----
    The HeadFile class provides simple ways to retrieve 2d and 3d
    head arrays from a MODFLOW binary head file and time series
    arrays for one or more cells.

    The BinaryLayerFile class is built on a record array consisting of
    headers, which are record arrays of the modflow header information
    (kstp, kper, pertim, totim, text, nrow, ncol, ilay)
    and long integers, which are pointers to first bytes of data for
    the corresponding data array.

    Examples
    --------

    >>> import flopy.utils.binaryfile as bf
    >>> hdobj = bf.HeadFile('model.hds', precision='single')
    >>> hdobj.list_records()
    >>> rec = hdobj.get_data(kstpkper=(1, 50))

    >>> ddnobj = bf.HeadFile('model.ddn', text='drawdown', precision='single')
    >>> ddnobj.list_records()
    >>> rec = ddnobj.get_data(totim=100.)


    """

    def __init__(
        self, filename, text="head", precision="auto", verbose=False, **kwargs
    ):
        self.text = text.encode()
        if precision == "auto":
            precision = get_headfile_precision(filename)
            if precision == "unknown":
                s = f"Error. Precision could not be determined for {filename}"
                print(s)
                raise Exception()
        self.header_dtype = BinaryHeader.set_dtype(
            bintype="Head", precision=precision
        )
        super().__init__(filename, precision, verbose, kwargs)
        return


class UcnFile(BinaryLayerFile):
    """
    UcnFile Class.

    Parameters
    ----------
    filename : string
        Name of the concentration file
    text : string
        Name of the text string in the ucn file.  Default is 'CONCENTRATION'
    precision : string
        'auto', 'single' or 'double'.  Default is 'auto'.
    verbose : bool
        Write information to the screen.  Default is False.

    Attributes
    ----------

    Methods
    -------

    See Also
    --------

    Notes
    -----
    The UcnFile class provides simple ways to retrieve 2d and 3d
    concentration arrays from a MT3D binary head file and time series
    arrays for one or more cells.

    The BinaryLayerFile class is built on a record array consisting of
    headers, which are record arrays of the modflow header information
    (kstp, kper, pertim, totim, text, nrow, ncol, ilay)
    and long integers, which are pointers to first bytes of data for
    the corresponding data array.

    Examples
    --------

    >>> import flopy.utils.binaryfile as bf
    >>> ucnobj = bf.UcnFile('MT3D001.UCN', precision='single')
    >>> ucnobj.list_records()
    >>> rec = ucnobj.get_data(kstpkper=(1,1))

    """

    def __init__(
        self,
        filename,
        text="concentration",
        precision="auto",
        verbose=False,
        **kwargs,
    ):
        self.text = text.encode()
        if precision == "auto":
            precision = get_headfile_precision(filename)
        if precision == "unknown":
            s = f"Error. Precision could not be determined for {filename}"
            print(s)
            raise Exception()
        self.header_dtype = BinaryHeader.set_dtype(
            bintype="Ucn", precision=precision
        )
        super().__init__(filename, precision, verbose, kwargs)
        return


class BudgetIndexError(Exception):
    pass

class CellBudgetFile:
    """
    The CellBudgetFile class provides convenient ways to retrieve and
    manipulate budget data from a binary cell budget file. A utility
    method is also provided to reverse the budget records for particle
    tracking simulations in which particles are tracked backwards from
    terminating to release locations (e.g., to compute capture zones).

    Parameters
    ----------
    filename : str or PathLike
        Path of the cell budget file.
    precision : string
        Precision of floating point budget data in the file. Accepted
        values are 'single' or 'double'. Default is 'single'.
    verbose : bool
        Toggle logging output. Default is False.

    Examples
    --------

    >>> import flopy.utils.binaryfile as bf
    >>> cbb = bf.CellBudgetFile('mymodel.cbb')
    >>> cbb.list_records()
    >>> rec = cbb.get_data(kstpkper=(0,0), text='RIVER LEAKAGE')

    """

    def __init__(
        self,
        filename: Union[str, os.PathLike],
        precision="auto",
        verbose=False,
        **kwargs,
    ):
        self.filename = Path(filename).expanduser().absolute()
        self.precision = precision
        self.verbose = verbose
        self.file = open(self.filename, "rb")
        # Get filesize to ensure this is not an empty file
        self.file.seek(0, 2)
        totalbytes = self.file.tell()
        self.file.seek(0, 0)  # reset to beginning
        assert self.file.tell() == 0
        if totalbytes == 0:
            raise ValueError(f"datafile error: file is empty: {filename}")
        self.nrow = 0
        self.ncol = 0
        self.nlay = 0
        self.nper = 0
        self.times = []
        self.kstpkper = []
        self.recordarray = []
        self.iposheader = []
        self.iposarray = []
        self.textlist = []
        self.imethlist = []
        self.paknamlist = []
        self.nrecords = 0
        self.compact = True  # compact budget file flag

        self.dis = None
        self.modelgrid = None
        if "model" in kwargs.keys():
            self.model = kwargs.pop("model")
            self.modelgrid = self.model.modelgrid
            self.dis = self.model.dis
        if "dis" in kwargs.keys():
            self.dis = kwargs.pop("dis")
            self.modelgrid = self.dis.parent.modelgrid
        if "tdis" in kwargs.keys():
            self.tdis = kwargs.pop("tdis")
        if "modelgrid" in kwargs.keys():
            self.modelgrid = kwargs.pop("modelgrid")
        if len(kwargs.keys()) > 0:
            args = ",".join(kwargs.keys())
            raise Exception(f"LayerFile error: unrecognized kwargs: {args}")

        if precision == "auto":
            success = self._set_precision("single")
            if not success:
                success = self._set_precision("double")
            if not success:
                s = "Budget precision could not be auto determined"
                raise BudgetIndexError(s)
        elif precision == "single":
            success = self._set_precision(precision)
        elif precision == "double":
            success = self._set_precision(precision)
        else:
            raise Exception(f"Unknown precision specified: {precision}")

        # set shape for full3D option
        if self.modelgrid is None:
            self.shape = (self.nlay, self.nrow, self.ncol)
            self.nnodes = self.nlay * self.nrow * self.ncol
        else:
            self.shape = self.modelgrid.shape
            self.nnodes = self.modelgrid.nnodes

        if not success:
            raise Exception(
                f"Budget file could not be read using {precision} precision"
            )

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __reset(self):
        """
        Reset indexing lists when determining precision
        """
        self.file.seek(0, 0)
        self.times = []
        self.kstpkper = []
        self.recordarray = []
        self.iposheader = []
        self.iposarray = []
        self.textlist = []
        self.imethlist = []
        self.paknamlist = []
        self.nrecords = 0

    def _set_precision(self, precision="single"):
        """
        Method to set the budget precsion from a CBC file. Enables
        Auto precision code to work

        Parameters
        ----------
        precision : str
            budget file precision (accepts 'single' or 'double')
        """
        success = True
        h1dt = [
            ("kstp", "i4"),
            ("kper", "i4"),
            ("text", "S16"),
            ("ncol", "i4"),
            ("nrow", "i4"),
            ("nlay", "i4"),
        ]
        if precision == "single":
            self.realtype = np.float32
            ffmt = "f4"
        else:
            self.realtype = np.float64
            ffmt = "f8"

        h2dt0 = [
            ("imeth", "i4"),
            ("delt", ffmt),
            ("pertim", ffmt),
            ("totim", ffmt),
        ]
        h2dt = [
            ("imeth", "i4"),
            ("delt", ffmt),
            ("pertim", ffmt),
            ("totim", ffmt),
            ("modelnam", "S16"),
            ("paknam", "S16"),
            ("modelnam2", "S16"),
            ("paknam2", "S16"),
        ]
        self.header1_dtype = np.dtype(h1dt)
        self.header2_dtype0 = np.dtype(h2dt0)
        self.header2_dtype = np.dtype(h2dt)
        hdt = h1dt + h2dt
        self.header_dtype = np.dtype(hdt)

        try:
            self._build_index()
        except BudgetIndexError:
            success = False
            self.__reset()

        return success

    def _totim_from_kstpkper(self, kstpkper):
        if self.dis is None:
            return -1.0
        kstp, kper = kstpkper
        perlen = self.dis.perlen.array
        nstp = self.dis.nstp.array[kper]
        tsmult = self.dis.tsmult.array[kper]
        kper_len = np.sum(perlen[:kper])
        this_perlen = perlen[kper]
        if tsmult == 1:
            dt1 = this_perlen / float(nstp)
        else:
            dt1 = this_perlen * (tsmult - 1.0) / ((tsmult**nstp) - 1.0)
        kstp_len = [dt1]
        for i in range(kstp + 1):
            kstp_len.append(kstp_len[-1] * tsmult)
        # kstp_len = np.array(kstp_len)
        # kstp_len = kstp_len[:kstp].sum()
        kstp_len = sum(kstp_len[: kstp + 1])
        return kper_len + kstp_len

    def _build_index(self):
        """
        Build the ordered dictionary, which maps the header information
        to the position in the binary file.
        """
        asciiset = " "
        for i in range(33, 127):
            asciiset += chr(i)

        # read first record
        header = self._get_header()
        nrow = header["nrow"]
        ncol = header["ncol"]
        text = header["text"]
        if isinstance(text, bytes):
            text = text.decode()
        if nrow < 0 or ncol < 0:
            raise Exception("negative nrow, ncol")
        if not text.endswith("FLOW-JA-FACE"):
            self.nrow = nrow
            self.ncol = ncol
            self.nlay = np.abs(header["nlay"])
        self.file.seek(0, 2)
        self.totalbytes = self.file.tell()
        self.file.seek(0, 0)
        self.recorddict = {}
        # read the remaining records
        ipos = 0
        while ipos < self.totalbytes:
            self.iposheader.append(ipos)
            header = self._get_header()
            self.nrecords += 1
            totim = header["totim"]
            # if old-style (non-compact) file,
            # compute totim from kstp and kper
            if not self.compact:
                totim = self._totim_from_kstpkper(
                    (header["kstp"] - 1, header["kper"] - 1)
                )
                header["totim"] = totim
            if totim >= 0 and totim not in self.times:
                self.times.append(totim)
            kstpkper = (header["kstp"], header["kper"])
            if kstpkper not in self.kstpkper:
                self.kstpkper.append(kstpkper)
            if header["text"] not in self.textlist:
                # check the precision of the file using text records
                try:
                    tlist = [header["text"], header["modelnam"]]
                    for text in tlist:
                        if isinstance(text, bytes):
                            text = text.decode()
                        for t in text:
                            if t.upper() not in asciiset:
                                raise Exception()

                except:
                    raise BudgetIndexError("Improper precision")
                self.textlist.append(header["text"])
                self.imethlist.append(header["imeth"])
            if header["paknam"] not in self.paknamlist:
                self.paknamlist.append(header["paknam"])
            ipos = self.file.tell()

            if self.verbose:
                for itxt in [
                    "kstp",
                    "kper",
                    "text",
                    "ncol",
                    "nrow",
                    "nlay",
                    "imeth",
                    "delt",
                    "pertim",
                    "totim",
                    "modelnam",
                    "paknam",
                    "modelnam2",
                    "paknam2",
                ]:
                    s = header[itxt]
                    if isinstance(s, bytes):
                        s = s.decode()
                    print(f"{itxt}: {s}")
                print("file position: ", ipos)
                if (
                    header["imeth"].item() != 5
                    and header["imeth"].item() != 6
                    and header["imeth"].item() != 7
                ):
                    print("")

            # set the nrow, ncol, and nlay if they have not been set
            if self.nrow == 0:
                text = header["text"]
                if isinstance(text, bytes):
                    text = text.decode()
                if not text.endswith("FLOW-JA-FACE"):
                    self.nrow = header["nrow"]
                    self.ncol = header["ncol"]
                    self.nlay = np.abs(header["nlay"])

            # store record and byte position mapping
            self.recorddict[
                tuple(header)
            ] = ipos  # store the position right after header2
            self.recordarray.append(header)
            self.iposarray.append(
                ipos
            )  # store the position right after header2

            # skip over the data to the next record and set ipos
            self._skip_record(header)
            ipos = self.file.tell()

        # convert to numpy arrays
        self.recordarray = np.array(self.recordarray, dtype=self.header_dtype)
        self.iposheader = np.array(self.iposheader, dtype=np.int64)
        self.iposarray = np.array(self.iposarray, dtype=np.int64)
        self.nper = self.recordarray["kper"].max()

    def _skip_record(self, header):
        """
        Skip over this record, not counting header and header2.

        """
        nlay = abs(header["nlay"])
        nrow = header["nrow"]
        ncol = header["ncol"]
        imeth = header["imeth"]
        if imeth == 0:
            nbytes = nrow * ncol * nlay * self.realtype(1).nbytes
        elif imeth == 1:
            nbytes = nrow * ncol * nlay * self.realtype(1).nbytes
        elif imeth == 2:
            nlist = binaryread(self.file, np.int32)[0]
            nbytes = nlist * (np.int32(1).nbytes + self.realtype(1).nbytes)
        elif imeth == 3:
            nbytes = nrow * ncol * self.realtype(1).nbytes
            nbytes += nrow * ncol * np.int32(1).nbytes
        elif imeth == 4:
            nbytes = nrow * ncol * self.realtype(1).nbytes
        elif imeth == 5:
            nauxp1 = binaryread(self.file, np.int32)[0]
            naux = nauxp1 - 1

            for i in range(naux):
                temp = binaryread(self.file, str, charlen=16)
            nlist = binaryread(self.file, np.int32)[0]
            if self.verbose:
                print("naux: ", naux)
                print("nlist: ", nlist)
                print("")
            nbytes = nlist * (
                np.int32(1).nbytes
                + self.realtype(1).nbytes
                + naux * self.realtype(1).nbytes
            )
        elif imeth == 6:
            # read rest of list data
            nauxp1 = binaryread(self.file, np.int32)[0]
            naux = nauxp1 - 1

            for i in range(naux):
                temp = binaryread(self.file, str, charlen=16)
            nlist = binaryread(self.file, np.int32)[0]
            if self.verbose:
                print("naux: ", naux)
                print("nlist: ", nlist)
                print("")
            nbytes = nlist * (
                np.int32(1).nbytes * 2
                + self.realtype(1).nbytes
                + naux * self.realtype(1).nbytes
            )
        else:
            raise Exception(f"invalid method code {imeth}")
        if nbytes != 0:
            self.file.seek(nbytes, 1)

    def _get_header(self):
        """
        Read the file header

        """
        header1 = binaryread(self.file, self.header1_dtype, (1,))
        nlay = header1["nlay"]
        self.compact = bool(nlay < 0)
        if self.compact:
            # fill header2 by first reading imeth, delt, pertim and totim
            # and then adding modelnames and paknames if imeth = 6
            temp = binaryread(self.file, self.header2_dtype0, (1,))
            header2 = np.array(
                [(0, 0.0, 0.0, 0.0, "", "", "", "")], dtype=self.header2_dtype
            )
            for name in temp.dtype.names:
                header2[name] = temp[name]
            if header2["imeth"].item() == 6:
                header2["modelnam"] = binaryread(self.file, str, charlen=16)
                header2["paknam"] = binaryread(self.file, str, charlen=16)
                header2["modelnam2"] = binaryread(self.file, str, charlen=16)
                header2["paknam2"] = binaryread(self.file, str, charlen=16)
        else:
            header2 = np.array(
                [(0, 0.0, 0.0, 0.0, "", "", "", "")], dtype=self.header2_dtype
            )
        fullheader = join_struct_arrays([header1, header2])
        return fullheader[0]

    def _find_text(self, text):
        """
        Determine if selected record name is in budget file

        """
        # check and make sure that text is in file
        text16 = None
        if text is not None:
            if isinstance(text, bytes):
                ttext = text.decode()
            else:
                ttext = text
            for t in self.textlist:
                if ttext.upper() in t.decode():
                    text16 = t
                    break
            if text16 is None:
                errmsg = "The specified text string is not in the budget file."
                raise Exception(errmsg)
        return text16

    def _find_paknam(self, paknam):
        """
        Determine if selected record name is in budget file

        """
        # check and make sure that text is in file
        paknam16 = None
        if paknam is not None:
            if isinstance(paknam, bytes):
                tpaknam = paknam.decode()
            else:
                tpaknam = paknam
            for t in self._unique_package_names():
                if tpaknam.upper() in t.decode():
                    paknam16 = t
                    break
            if paknam16 is None:
                raise Exception(
                    "The specified package name string is not "
                    "in the budget file."
                )
        return paknam16

    def list_records(self):
        """
        Print a list of all of the records in the file
        """
        for rec in self.recordarray:
            if isinstance(rec, bytes):
                rec = rec.decode()
            print(rec)

    def list_unique_records(self):
        """
        Print a list of unique record names
        """
        print("RECORD           IMETH")
        print(22 * "-")
        for rec, imeth in zip(self.textlist, self.imethlist):
            if isinstance(rec, bytes):
                rec = rec.decode()
            print(f"{rec.strip():16} {imeth:5d}")

    def list_unique_packages(self):
        """
        Print a list of unique package names
        """
        for rec in self._unique_package_names():
            if isinstance(rec, bytes):
                rec = rec.decode()
            print(rec)

    def get_unique_record_names(self, decode=False):
        """
        Get a list of unique record names in the file

        Parameters
        ----------
        decode : bool
            Optional boolean used to decode byte strings (default is False).

        Returns
        ----------
        names : list of strings
            List of unique text names in the binary file.

        """
        if decode:
            names = []
            for text in self.textlist:
                if isinstance(text, bytes):
                    text = text.decode()
                names.append(text)
        else:
            names = self.textlist
        return names

    def get_unique_package_names(self, decode=False):
        """
        Get a list of unique package names in the file

        Parameters
        ----------
        decode : bool
            Optional boolean used to decode byte strings (default is False).

        Returns
        ----------
        names : list of strings
            List of unique package names in the binary file.

        """
        if decode:
            names = []
            for text in self.paknamlist:
                if isinstance(text, bytes):
                    text = text.decode()
                names.append(text)
        else:
            names = self.paknamlist
        return names

    def _unique_package_names(self):
        """
        Get a list of unique package names in the file

        Returns
        ----------
        out : list of strings
            List of unique package names in the binary file.

        """
        return self.paknamlist

    def get_kstpkper(self):
        """
        Get a list of unique stress periods and time steps in the file

        Returns
        ----------
        out : list of (kstp, kper) tuples
            List of unique kstp, kper combinations in binary file.  kstp and
            kper values are zero-based.

        """
        kstpkper = []
        for kstp, kper in self.kstpkper:
            kstpkper.append((kstp - 1, kper - 1))
        return kstpkper

    def get_indices(self, text=None):
        """
        Get a list of indices for a selected record name

        Parameters
        ----------
        text : str
            The text identifier for the record.  Examples include
            'RIVER LEAKAGE', 'STORAGE', 'FLOW RIGHT FACE', etc.

        Returns
        ----------
        out : tuple
            indices of selected record name in budget file.

        """
        # check and make sure that text is in file
        if text is not None:
            text16 = self._find_text(text)
            select_indices = np.where(self.recordarray["text"] == text16)
            if isinstance(select_indices, tuple):
                select_indices = select_indices[0]
        else:
            select_indices = None
        return select_indices

    def get_position(self, idx, header=False):
        """
        Get the starting position of the data or header for a specified record
        number in the binary budget file.

        Parameters
        ----------
        idx : int
            The zero-based record number.  The first record is record 0.
        header : bool
            If True, the position of the start of the header data is returned.
            If False, the position of the start of the data is returned
            (default is False).

        Returns
        -------
        ipos : int64
            The position of the start of the data in the cell budget file
            or the start of the header.

        """
        if header:
            ipos = self.iposheader[idx]
        else:
            ipos = self.iposarray[idx]
        return ipos

    def get_data(
        self,
        idx=None,
        kstpkper=None,
        totim=None,
        text=None,
        paknam=None,
        full3D=False,
    ):
        """
        Get data from the binary budget file.

        Parameters
        ----------
        idx : int or list
            The zero-based record number.  The first record is record 0.
        kstpkper : tuple of ints
            A tuple containing the time step and stress period (kstp, kper).
            The kstp and kper values are zero based.
        totim : float
            The simulation time.
        text : str
            The text identifier for the record.  Examples include
            'RIVER LEAKAGE', 'STORAGE', 'FLOW RIGHT FACE', etc.
        full3D : boolean
            If true, then return the record as a three dimensional numpy
            array, even for those list-style records written as part of a
            'COMPACT BUDGET' MODFLOW budget file.  (Default is False.)

        Returns
        ----------
        recordlist : list of records
            A list of budget objects.  The structure of the returned object
            depends on the structure of the data in the cbb file.

            If full3D is True, then this method will return a numpy masked
            array of size (nlay, nrow, ncol) for those list-style
            'COMPACT BUDGET' records written by MODFLOW.

        See Also
        --------

        Notes
        -----

        Examples
        --------

        """
        # trap for totim error
        if totim is not None:
            if len(self.times) == 0:
                errmsg = """This is an older style budget file that
                         does not have times in it.  Use the MODFLOW 
                         compact budget format if you want to work with 
                         times.  Or you may access this file using the
                         kstp and kper arguments or the idx argument."""
                raise Exception(errmsg)

        # check and make sure that text is in file
        text16 = None
        if text is not None:
            text16 = self._find_text(text)
        paknam16 = None
        if paknam is not None:
            paknam16 = self._find_paknam(paknam)

        if kstpkper is not None:
            kstp1 = kstpkper[0] + 1
            kper1 = kstpkper[1] + 1
            if text is None and paknam is None:
                select_indices = np.where(
                    (self.recordarray["kstp"] == kstp1)
                    & (self.recordarray["kper"] == kper1)
                )
            else:
                if paknam is None and text is not None:
                    select_indices = np.where(
                        (self.recordarray["kstp"] == kstp1)
                        & (self.recordarray["kper"] == kper1)
                        & (self.recordarray["text"] == text16)
                    )
                elif text is None and paknam is not None:
                    select_indices = np.where(
                        (self.recordarray["kstp"] == kstp1)
                        & (self.recordarray["kper"] == kper1)
                        & (self.recordarray["paknam"] == paknam16)
                    )
                else:
                    select_indices = np.where(
                        (self.recordarray["kstp"] == kstp1)
                        & (self.recordarray["kper"] == kper1)
                        & (self.recordarray["text"] == text16)
                        & (self.recordarray["paknam"] == paknam16)
                    )

        elif totim is not None:
            if text is None and paknam is None:
                select_indices = np.where(self.recordarray["totim"] == totim)
            else:
                if paknam is None and text is not None:
                    select_indices = np.where(
                        (self.recordarray["totim"] == totim)
                        & (self.recordarray["text"] == text16)
                    )
                elif text is None and paknam is not None:
                    select_indices = np.where(
                        (self.recordarray["totim"] == totim)
                        & (self.recordarray["paknam"] == paknam16)
                    )
                else:
                    select_indices = np.where(
                        (self.recordarray["totim"] == totim)
                        & (self.recordarray["text"] == text16)
                        & (self.recordarray["paknam"] == paknam16)
                    )

        # allow for idx to be a list or a scalar
        elif idx is not None:
            if isinstance(idx, list):
                select_indices = idx
            else:
                select_indices = [idx]

        # case where only text is entered
        elif text is not None:
            select_indices = np.where(self.recordarray["text"] == text16)

        else:
            raise TypeError(
                "get_data() missing 1 required argument: 'kstpkper', 'totim', "
                "'idx', or 'text'"
            )

        # build and return the record list
        if isinstance(select_indices, tuple):
            select_indices = select_indices[0]
        recordlist = []
        for idx in select_indices:
            rec = self.get_record(idx, full3D=full3D)
            recordlist.append(rec)

        return recordlist

    def get_ts(self, idx, text=None, times=None):
        """
        Get a time series from the binary budget file.

        Parameters
        ----------
        idx : tuple of ints, or a list of a tuple of ints
            idx can be (layer, row, column) or it can be a list in the form
            [(layer, row, column), (layer, row, column), ...].  The layer,
            row, and column values must be zero based.
        text : str
            The text identifier for the record.  Examples include
            'RIVER LEAKAGE', 'STORAGE', 'FLOW RIGHT FACE', etc.
        times : iterable of floats
            List of times to from which to get time series.

        Returns
        ----------
        out : numpy array
            Array has size (ntimes, ncells + 1).  The first column in the
            data array will contain time (totim).

        See Also
        --------

        Notes
        -----

        The layer, row, and column values must be zero-based, and must be
        within the following ranges: 0 <= k < nlay; 0 <= i < nrow; 0 <= j < ncol

        Examples
        --------

        """
        # issue exception if text not provided
        if text is None:
            raise Exception(
                "text keyword must be provided to CellBudgetFile "
                "get_ts() method."
            )

        kijlist = self._build_kijlist(idx)
        nstation = self._get_nstation(idx, kijlist)

        # Initialize result array and put times in first column
        result = self._init_result(nstation)

        kk = self.get_kstpkper()
        timesint = self.get_times()
        if len(timesint) < 1:
            if times is None:
                timesint = [x + 1 for x in range(len(kk))]
            else:
                if isinstance(times, np.ndarray):
                    times = times.tolist()
                if len(times) != len(kk):
                    raise Exception(
                        "times passed to CellBudgetFile get_ts() "
                        "method must be equal to {} "
                        "not {}".format(len(kk), len(times))
                    )
                timesint = times
        for idx, t in enumerate(timesint):
            result[idx, 0] = t

        for itim, k in enumerate(kk):
            try:
                v = self.get_data(kstpkper=k, text=text, full3D=True)
                # skip missing data - required for storage
                if len(v) > 0:
                    v = v[0]
                    istat = 1
                    for k, i, j in kijlist:
                        result[itim, istat] = v[k, i, j].copy()
                        istat += 1
            except ValueError:
                v = self.get_data(kstpkper=k, text=text)
                # skip missing data - required for storage
                if len(v) > 0:
                    if self.modelgrid is None:
                        s = (
                            "A modelgrid instance must be provided during "
                            "instantiation to get IMETH=6 timeseries data"
                        )
                        raise AssertionError(s)

                    if self.modelgrid.grid_type == "structured":
                        ndx = [
                            lrc[0]
                            * (self.modelgrid.nrow * self.modelgrid.ncol)
                            + lrc[1] * self.modelgrid.ncol
                            + (lrc[2] + 1)
                            for lrc in kijlist
                        ]
                    else:
                        ndx = [
                            lrc[0] * self.modelgrid.ncpl + (lrc[-1] + 1)
                            for lrc in kijlist
                        ]

                    for vv in v:
                        field = vv.dtype.names[2]
                        dix = np.where(np.isin(vv["node"], ndx))[0]
                        if len(dix) > 0:
                            result[itim, 1:] = vv[field][dix]

        return result

    def _build_kijlist(self, idx):
        if isinstance(idx, list):
            kijlist = idx
        elif isinstance(idx, tuple):
            kijlist = [idx]
        else:
            raise Exception("Could not build kijlist from ", idx)

        # Check to make sure that k, i, j are within range, otherwise
        # the seek approach won't work.  Can't use k = -1, for example.
        for k, i, j in kijlist:
            fail = False
            if k < 0 or k > self.nlay - 1:
                fail = True
            if i < 0 or i > self.nrow - 1:
                fail = True
            if j < 0 or j > self.ncol - 1:
                fail = True
            if fail:
                raise Exception(
                    "Invalid cell index. Cell {} not within model grid: "
                    "{}".format((k, i, j), (self.nlay, self.nrow, self.ncol))
                )
        return kijlist

    def _get_nstation(self, idx, kijlist):
        if isinstance(idx, list):
            return len(kijlist)
        elif isinstance(idx, tuple):
            return 1

    def _init_result(self, nstation):
        # Initialize result array and put times in first column
        result = np.empty(
            (len(self.kstpkper), nstation + 1), dtype=self.realtype
        )
        result[:, :] = np.nan
        if len(self.times) == result.shape[0]:
            result[:, 0] = np.array(self.times)
        return result

    def get_record(self, idx, full3D=False):
        """
        Get a single data record from the budget file.

        Parameters
        ----------
        idx : int
            The zero-based record number.  The first record is record 0.
        full3D : boolean
            If true, then return the record as a three dimensional numpy
            array, even for those list-style records written as part of a
            'COMPACT BUDGET' MODFLOW budget file.  (Default is False.)

        Returns
        ----------
        record : a single data record
            The structure of the returned object depends on the structure of
            the data in the cbb file. Compact list data are returned as

            If full3D is True, then this method will return a numpy masked
            array of size (nlay, nrow, ncol) for those list-style
            'COMPACT BUDGET' records written by MODFLOW.

        See Also
        --------

        Notes
        -----

        Examples
        --------

        """
        # idx must be an ndarray, so if it comes in as an integer then convert
        if np.isscalar(idx):
            idx = np.array([idx])

        header = self.recordarray[idx]
        ipos = self.iposarray[idx].item()
        self.file.seek(ipos, 0)
        imeth = header["imeth"][0]

        t = header["text"][0]
        if isinstance(t, bytes):
            t = t.decode("utf-8")
        s = f"Returning {t.strip()} as "

        nlay = abs(header["nlay"][0])
        nrow = header["nrow"][0]
        ncol = header["ncol"][0]

        # default method
        if imeth == 0:
            if self.verbose:
                s += f"an array of shape {(nlay, nrow, ncol)}"
                print(s)
            return binaryread(
                self.file, self.realtype(1), shape=(nlay, nrow, ncol)
            )
        # imeth 1
        elif imeth == 1:
            if self.verbose:
                s += f"an array of shape {(nlay, nrow, ncol)}"
                print(s)
            return binaryread(
                self.file, self.realtype(1), shape=(nlay, nrow, ncol)
            )

        # imeth 2
        elif imeth == 2:
            nlist = binaryread(self.file, np.int32)[0]
            dtype = np.dtype([("node", np.int32), ("q", self.realtype)])
            if self.verbose:
                if full3D:
                    s += (
                        f"a numpy masked array of "
                        f"size ({nlay}, {nrow}, {ncol})"
                    )
                else:
                    s += f"a numpy recarray of size ({nlist}, 2)"
                print(s)
            data = binaryread(self.file, dtype, shape=(nlist,))
            if full3D:
                return self.__create3D(data)
            else:
                return data.view(np.recarray)

        # imeth 3
        elif imeth == 3:
            ilayer = binaryread(self.file, np.int32, shape=(nrow, ncol))
            data = binaryread(self.file, self.realtype(1), shape=(nrow, ncol))
            if self.verbose:
                if full3D:
                    s += (
                        "a numpy masked array of size "
                        f"({nlay}, {nrow}, {ncol})"
                    )
                else:
                    s += (
                        "a list of two 2D numpy arrays. The first is an "
                        f"integer layer array of shape ({nrow}, {ncol}). The "
                        f"second is real data array of shape ({nrow}, {ncol})"
                    )
                print(s)
            if full3D:
                out = np.ma.zeros(self.nnodes, dtype=np.float32)
                out.mask = True
                vertical_layer = ilayer.flatten() - 1
                # create the 2D cell index and then move it to
                # the correct vertical location
                idx = np.arange(0, vertical_layer.shape[0])
                idx += vertical_layer * nrow * ncol
                out[idx] = data.flatten()
                return out.reshape(self.shape)
            else:
                return [ilayer, data]

        # imeth 4
        elif imeth == 4:
            if self.verbose:
                s += f"a 2d numpy array of size ({nrow}, {ncol})"
                print(s)
            return binaryread(self.file, self.realtype(1), shape=(nrow, ncol))

        # imeth 5
        elif imeth == 5:
            nauxp1 = binaryread(self.file, np.int32)[0]
            naux = nauxp1 - 1
            l = [("node", np.int32), ("q", self.realtype)]
            for i in range(naux):
                auxname = binaryread(self.file, str, charlen=16)
                if not isinstance(auxname, str):
                    auxname = auxname.decode()
                l.append((auxname.strip(), self.realtype))
            dtype = np.dtype(l)
            nlist = binaryread(self.file, np.int32)[0]
            data = binaryread(self.file, dtype, shape=(nlist,))
            if full3D:
                if self.verbose:
                    s += f"a list array of shape ({nlay}, {nrow}, {ncol})"
                    print(s)
                return self.__create3D(data)
            else:
                if self.verbose:
                    s += f"a numpy recarray of size ({nlist}, {2 + naux})"
                    print(s)
                return data.view(np.recarray)

        # imeth 6
        elif imeth == 6:
            # read rest of list data
            nauxp1 = binaryread(self.file, np.int32)[0]
            naux = nauxp1 - 1
            l = [("node", np.int32), ("node2", np.int32), ("q", self.realtype)]
            for i in range(naux):
                auxname = binaryread(self.file, str, charlen=16)
                if not isinstance(auxname, str):
                    auxname = auxname.decode()
                l.append((auxname.strip(), self.realtype))
            dtype = np.dtype(l)
            nlist = binaryread(self.file, np.int32)[0]
            data = binaryread(self.file, dtype, shape=(nlist,))
            if self.verbose:
                if full3D:
                    s += f"a list array of shape ({nlay}, {nrow}, {ncol})"
                else:
                    s += f"a numpy recarray of size ({nlist}, 2)"
                print(s)
            if full3D:
                data = self.__create3D(data)
                if self.modelgrid is not None:
                    return np.reshape(data, self.shape)
                else:
                    return data
            else:
                return data.view(np.recarray)
        else:
            raise ValueError(f"invalid imeth value - {imeth}")

        # should not reach this point
        return

    def __create3D(self, data):
        """
        Convert a dictionary of {node: q, ...} into a numpy masked array.
        Used to create full grid arrays when the full3D keyword is set
        to True in get_data.

        Parameters
        ----------
        data : dictionary
            Dictionary with node keywords and flows (q) items.

        Returns
        ----------
        out : numpy masked array
            List contains unique simulation times (totim) in binary file.

        """
        out = np.ma.zeros(self.nnodes, dtype=np.float32)
        out.mask = True
        for [node, q] in zip(data["node"], data["q"]):
            idx = node - 1
            out.data[idx] += q
            out.mask[idx] = False
        return np.ma.reshape(out, self.shape)

    def get_times(self):
        """
        Get a list of unique times in the file

        Returns
        ----------
        out : list of floats
            List contains unique simulation times (totim) in binary file.

        """
        return self.times

    def get_nrecords(self):
        """
        Return the number of records in the file

        Returns
        -------

        out : int
            Number of records in the file.

        """
        return self.recordarray.shape[0]

    def get_residual(self, totim, scaled=False):
        """
        Return an array the size of the model grid containing the flow residual
        calculated from the budget terms.  Residual will not be correct unless
        all flow terms are written to the budget file.

        Parameters
        ----------
        totim : float
            Simulation time for which to calculate the residual.  This value
            must be precise, so it is best to get it from the get_times
            method.

        scaled : bool
            If True, then divide the residual by the total cell inflow

        Returns
        -------
        residual : np.ndarray
            The flow residual for the cell of shape (nlay, nrow, ncol)

        """

        nlay = self.nlay
        nrow = self.nrow
        ncol = self.ncol
        residual = np.zeros((nlay, nrow, ncol), dtype=float)
        if scaled:
            inflow = np.zeros((nlay, nrow, ncol), dtype=float)
        select_indices = np.where(self.recordarray["totim"] == totim)[0]

        for i in select_indices:
            text = self.recordarray[i]["text"].decode()
            if self.verbose:
                print(f"processing {text}")
            flow = self.get_record(idx=i, full3D=True)
            if ncol > 1 and "RIGHT FACE" in text:
                residual -= flow[:, :, :]
                residual[:, :, 1:] += flow[:, :, :-1]
                if scaled:
                    idx = np.where(flow < 0.0)
                    inflow[idx] -= flow[idx]
                    idx = np.where(flow > 0.0)
                    l, r, c = idx
                    idx = (l, r, c + 1)
                    inflow[idx] += flow[idx]
            elif nrow > 1 and "FRONT FACE" in text:
                residual -= flow[:, :, :]
                residual[:, 1:, :] += flow[:, :-1, :]
                if scaled:
                    idx = np.where(flow < 0.0)
                    inflow[idx] -= flow[idx]
                    idx = np.where(flow > 0.0)
                    l, r, c = idx
                    idx = (l, r + 1, c)
                    inflow[idx] += flow[idx]
            elif nlay > 1 and "LOWER FACE" in text:
                residual -= flow[:, :, :]
                residual[1:, :, :] += flow[:-1, :, :]
                if scaled:
                    idx = np.where(flow < 0.0)
                    inflow[idx] -= flow[idx]
                    idx = np.where(flow > 0.0)
                    l, r, c = idx
                    idx = (l + 1, r, c)
                    inflow[idx] += flow[idx]
            else:
                residual += flow
                if scaled:
                    idx = np.where(flow > 0.0)
                    inflow[idx] += flow[idx]

        if scaled:
            residual_scaled = np.zeros((nlay, nrow, ncol), dtype=float)
            idx = inflow > 0.0
            residual_scaled[idx] = residual[idx] / inflow[idx]
            return residual_scaled

        return residual

    def close(self):
        """
        Close the file handle
        """
        self.file.close()

    def reverse(self, filename: Optional[os.PathLike] = None):
        """
        Write a binary cell budget file with the records in reverse order.
        If a new filename is not provided, or if the filename is the same
        as the existing filename, the file will be overwritten and data
        reloaded from the rewritten/reversed file.

        Parameters
        ----------

        filename : str or PathLike, optional
            Path of the new reversed binary cell budget file to create.
        """

        filename = (
            Path(filename).expanduser().absolute()
            if filename
            else self.filename
        )

        # header array formats
        dt1 = np.dtype(
            [
                ("kstp", np.int32),
                ("kper", np.int32),
                ("text", "S16"),
                ("ndim1", np.int32),
                ("ndim2", np.int32),
                ("ndim3", np.int32),
                ("imeth", np.int32),
                ("delt", np.float64),
                ("pertim", np.float64),
                ("totim", np.float64),
            ]
        )
        dt2 = np.dtype(
            [
                ("text1id1", "S16"),
                ("text1id2", "S16"),
                ("text2id1", "S16"),
                ("text2id2", "S16"),
            ]
        )

        # make sure we have tdis
        if self.tdis is None or not any(self.tdis.perioddata.get_data()):
            raise ValueError(
                "tdis must be known to reverse a cell budget file"
            )

        # extract perioddata
        pd = self.tdis.perioddata.get_data()

        # get maximum period number and total simulation time
        nper = len(pd)
        kpermx = nper - 1
        tsimtotal = 0.0
        for tpd in pd:
            tsimtotal += tpd[0]

        # get number of records
        nrecords = self.get_nrecords()

        # open backward budget file
        with open(filename, "wb") as fbin:
            # loop over budget file records in reverse order
            for idx in range(nrecords - 1, -1, -1):
                # load header array
                header = self.recordarray[idx]

                # reverse kstp and kper in the header array
                (kstp, kper) = (header["kstp"] - 1, header["kper"] - 1)
                kstpmx = pd[kper][1] - 1
                kstpb = kstpmx - kstp
                kperb = kpermx - kper
                (header["kstp"], header["kper"]) = (kstpb + 1, kperb + 1)

                # reverse totim and pertim in the header array
                header["totim"] = tsimtotal - header["totim"]
                perlen = pd[kper][0]
                header["pertim"] = perlen - header["pertim"]

                # Write main header information to backward budget file
                h = header[
                    [
                        "kstp",
                        "kper",
                        "text",
                        "ncol",
                        "nrow",
                        "nlay",
                        "imeth",
                        "delt",
                        "pertim",
                        "totim",
                    ]
                ]
                # Note: much of the code below is based on binary_file_writer.py
                h = np.array(h, dtype=dt1)
                h.tofile(fbin)
                if header["imeth"] == 6:
                    # Write additional header information to the backward budget file
                    h = header[
                        [
                            "modelnam",
                            "paknam",
                            "modelnam2",
                            "paknam2",
                        ]
                    ]
                    h = np.array(h, dtype=dt2)
                    h.tofile(fbin)
                    # Load data
                    data = self.get_data(idx)[0]
                    data = np.array(data)
                    # Negate flows
                    data["q"] = -data["q"]
                    # Write ndat (number of floating point columns)
                    colnames = data.dtype.names
                    ndat = len(colnames) - 2
                    dt = np.dtype([("ndat", np.int32)])
                    h = np.array([(ndat,)], dtype=dt)
                    h.tofile(fbin)
                    # Write auxiliary column names
                    naux = ndat - 1
                    if naux > 0:
                        auxtxt = [
                            "{:16}".format(colname) for colname in colnames[3:]
                        ]
                        auxtxt = tuple(auxtxt)
                        dt = np.dtype(
                            [(colname, "S16") for colname in colnames[3:]]
                        )
                        h = np.array(auxtxt, dtype=dt)
                        h.tofile(fbin)
                    # Write nlist
                    nlist = data.shape[0]
                    dt = np.dtype([("nlist", np.int32)])
                    h = np.array([(nlist,)], dtype=dt)
                    h.tofile(fbin)
                elif header["imeth"] == 1:
                    # Load data
                    data = self.get_data(idx)[0][0][0]
                    data = np.array(data, dtype=np.float64)
                    # Negate flows
                    data = -data
                else:
                    raise ValueError("not expecting imeth " + header["imeth"])
                # Write data
                data.tofile(fbin)

        # if we rewrote the original file, reinitialize
        if filename == self.filename:
            self.__init__(self.filename, self.precision, self.verbose)

class HeadUFile(BinaryLayerFile):
    """
    Unstructured MODFLOW-USG HeadUFile Class.

    Parameters
    ----------
    filename : string
        Name of the concentration file
    text : string
        Name of the text string in the head file.  Default is 'headu'
    precision : string
        'auto', 'single' or 'double'.  Default is 'auto'.
    verbose : bool
        Write information to the screen.  Default is False.

    Attributes
    ----------

    Methods
    -------

    See Also
    --------

    Notes
    -----
    The HeadUFile class provides simple ways to retrieve a list of
    head arrays from a MODFLOW-USG binary head file and time series
    arrays for one or more cells.

    The BinaryLayerFile class is built on a record array consisting of
    headers, which are record arrays of the modflow header information
    (kstp, kper, pertim, totim, text, nrow, ncol, ilay)
    and long integers, which are pointers to first bytes of data for
    the corresponding data array.  For unstructured grids, nrow and ncol
    are the starting and ending node numbers for layer, ilay.  This class
    overrides methods in the parent class so that the proper sized arrays
    are created.

    When the get_data method is called for this class, a list of
    one-dimensional arrays will be returned, where each array is the head
    array for a layer.  If the heads for a layer were not saved, then
    None will be returned for that layer.

    Examples
    --------

    >>> import flopy.utils.binaryfile as bf
    >>> hdobj = bf.HeadUFile('model.hds')
    >>> hdobj.list_records()
    >>> usgheads = hdobj.get_data(kstpkper=(1, 50))


    """

    def __init__(
        self, filename, text="headu", precision="auto", verbose=False, **kwargs
    ):
        """
        Class constructor
        """
        self.text = text.encode()
        if precision == "auto":
            precision = get_headfile_precision(filename)
            if precision == "unknown":
                s = f"Error. Precision could not be determined for {filename}"
                print(s)
                raise Exception()
        self.header_dtype = BinaryHeader.set_dtype(
            bintype="Head", precision=precision
        )
        super().__init__(filename, precision, verbose, kwargs)
        return

    def _get_data_array(self, totim=0.0):
        """
        Get a list of 1D arrays for the
        specified kstp and kper value or totim value.

        """

        if totim >= 0.0:
            keyindices = np.where((self.recordarray["totim"] == totim))[0]
            if len(keyindices) == 0:
                msg = f"totim value ({totim}) not found in file..."
                raise Exception(msg)
        else:
            raise Exception("Data not found...")

        # fill a list of 1d arrays with heads from binary file
        data = self.nlay * [None]
        for idx in keyindices:
            ipos = self.iposarray[idx]
            ilay = self.recordarray["ilay"][idx]
            nstrt = self.recordarray["ncol"][idx]
            nend = self.recordarray["nrow"][idx]
            npl = nend - nstrt + 1
            if self.verbose:
                print(f"Byte position in file: {ipos} for layer {ilay}")
            self.file.seek(ipos, 0)
            data[ilay - 1] = binaryread(self.file, self.realtype, shape=(npl,))
        return data

    def get_databytes(self, header):
        """

        Parameters
        ----------
        header : datafile.Header
            header object

        Returns
        -------
         databytes : int
            size of the data array, in bytes, following the header

        """
        # unstructured head files contain node starting and ending indices
        # for each layer
        nstrt = np.int64(header["ncol"])
        nend = np.int64(header["nrow"])
        npl = nend - nstrt + 1
        return npl * np.int64(self.realtype(1).nbytes)

    def get_ts(self, idx):
        """
        Get a time series from the binary HeadUFile

        Parameters
        ----------
        idx : int or list of ints
            idx can be nodenumber or it can be a list in the form
            [nodenumber, nodenumber, ...].  The nodenumber,
            values must be zero based.

        Returns
        ----------
        out : numpy array
            Array has size (ntimes, ncells + 1).  The first column in the
            data array will contain time (totim).

        """
        times = self.get_times()

        # find node number in layer that node is in
        data = self.get_data(totim=times[0])
        nodelay = [len(data[lay]) for lay in range(len(data))]
        nodelay_cumsum = np.cumsum([0] + nodelay)

        if isinstance(idx, int):
            layer = np.searchsorted(nodelay_cumsum, idx)
            nnode = idx - nodelay_cumsum[layer - 1]

            result = []
            for i, time in enumerate(times):
                data = self.get_data(totim=time)
                result.append([time, data[layer - 1][nnode]])

        elif isinstance(idx, list):

            result = []
            for i, time in enumerate(times):
                data = self.get_data(totim=time)
                row = [time]

                for node in idx:
                    if isinstance(node, int):
                        layer = np.searchsorted(nodelay_cumsum, node)
                        nnode = node - nodelay_cumsum[layer - 1]
                        row += [data[layer - 1][nnode]]
                    else:
                        errmsg = "idx must be an integer or a list of integers"
                        raise Exception(errmsg)

                result.append(row)

        else:
            errmsg = "idx must be an integer or a list of integers"
            raise Exception(errmsg)

        return np.array(result)
