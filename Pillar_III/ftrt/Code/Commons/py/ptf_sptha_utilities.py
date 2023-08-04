
# Import system modules
import numpy as np

def countX(lst, x):
    count = 0
    for ele in lst:
        if (ele == x):
            count = count + 1
    return count

def region_type_number_2_array(**kwargs):
#generate array for seismicity types in regions ((while reading the regionalization input file)
    n = kwargs.get('numbers', None).replace(',','').split()
    results = [int(i) for i in n]
    return results

def region_coordinates_points_splitter_2_dictionary(**kwargs):
#generate arrays for coordinates of polygon delimiting regions (while reading the regionalization input file)

    n = kwargs.get('points', None).replace(',','').split()

    # Generate array of lon, lat vales that delimits the poligon
    # this is an np array
    lon_lat_sequence = np.array([float(i) for i in n])
    lls = np.split(lon_lat_sequence, int(len(lon_lat_sequence)/2))

    # generate an array of [lon, lat] couple for each point of the poligon
    # this is an np array
    polygon_sequence = []

    for i in range(len(lls)):
        #polygon_sequence.append(list(lls[i]))
        polygon_sequence.append(lls[i])
    polygon_sequence = np.array(polygon_sequence)

    # generate array of latitudes and longitudes
    Tlon = []
    Tlat  = []
    for i in range(len(polygon_sequence)):
        Tlon.append(polygon_sequence[i][0])
        Tlat.append(polygon_sequence[i][1])
    Tlon = np.array(Tlon)
    Tlat  = np.array(Tlat)

    #coordinates = {'lon_lat_sequence'    : lon_lat_sequence, \
    #           'vertex_sequence'     : polygon_sequence, \
    #           'Tlon' : Tlon, 'Tlat'  : Tlat }

#    return coordinates
    return len(polygon_sequence),polygon_sequence,Tlon,Tlat

def get_BS_discretizations(**kwargs):

    from shapely import geometry


    poly            = kwargs.get('regions_poly', None)
    IDReg           = kwargs.get('IDreg', None)
    #print(IDReg)
    #sys.exit()
    #moho_pos        = kwargs.get('mxy', None)
    moho_dep        = kwargs.get('mz', None)
    discretizations = kwargs.get('tmpdiscr', None)
    file_name       = kwargs.get('file_name', None)
    grid_moho       = kwargs.get('grid_moho', None)
    grid_discr      = kwargs.get('grid_discr', None)
    moho_all        = kwargs.get('moho_all', None)

    inode           = kwargs.get('i', None)
    #0: BS-1_Magnitude
    #1: BS-2_Position
    #2: BS-3_Depth
    #3: BS-4_FocalMechanism
    #4: BS-5_Area
    #5: BS-6_Slip

    #print(poly)
    #sys.exit()

    #open(file)
    f = open(file_name, "r")

    #print(inode)
    #print(file_name)
    #sys.exit()

    # Magnitude Discretization or FocaMech
    if (inode == 0):

        ID = []
        Val = []

        for lines in f:
            foe = lines.rstrip().split(':')

            ID.append(foe[0])
            Val.append(float(foe[1]))

        items = {'ID': ID,'Val': Val}

    elif (inode == 3):

        ID = []
        Val = []

        for lines in f:
            foe = lines.rstrip().split(':')

            ID.append(foe[0])
            Val.append(foe[1])

        items = {'ID': ID,'Val': Val}

    # Position Discretization
    elif (inode == 1):

        ID = []
        Val = []
        Val_x = []
        Val_y = []
        Region = []
        DepthMoho = []

        nr= 0

        # Cerca Regioni
        for lines in f:
            nr= 0
            foe = lines.rstrip().split(':')

            ID.append(foe[0])
            Val.append(foe[1])
            clon, clat = (float(x)  for x in foe[1].split())

            Val_x.append(clon)
            Val_y.append(clat)

            for rp in poly:
                nr += 1
                if (rp.contains(geometry.Point(clon, clat))):
                    Region.append(nr)
                    break

        # Cerca profondità moho
        for i in range(len(grid_discr)):

            # cerca coppia di coordinate (grid_discr[i]) nell'arrey di coppie della maho
            c = np.where(grid_moho == grid_discr[i])
            # Per qualche ragione da un duplicato di indici all'indice che trova. Quindi trova indice doppio
            m = np.zeros_like(c[0], dtype=bool)
            m[np.unique(c[0], return_index=True)[1]] = True
            # Indice della moho che si cerca è questo:
            index=c[0][~m][0]
            #raccogli tutt ele profondità
            DepthMoho.append(moho_dep[index])


        items =  {'ID': ID,'Val': Val, 'Region': Region, 'DepthMoho': DepthMoho, 'Val_x' : Val_x, 'Val_y': Val_y, 'grid_moho' : moho_all }

    elif (inode == 2):
        ID = []

        IDMag = np.array(discretizations['BS-1_Magnitude']['ID'])
        IDPos = np.array(discretizations['BS-2_Position']['ID'])

        ValVec = np.empty((len(IDMag),len(IDPos)),dtype=np.ndarray)
        ValVecNum = np.empty((len(IDMag),len(IDPos)))

        for lines in f:
            IDMagTmp = lines[0:4]
            IDPosTmp = lines[5:16]
            ValTmp = np.array(lines.rstrip().split(':')[1].split(',')).astype(np.float)

            TmpPosMag = np.argwhere(IDMag == IDMagTmp)[0][0]
            TmpPosPos = np.argwhere(IDPos == IDPosTmp)[0][0]
            ValVec[TmpPosMag,TmpPosPos] = ValTmp
            ValVecNum[TmpPosMag,TmpPosPos] = len(ValTmp)

        items =  {'ID': ID, 'ValVec': ValVec, 'ValVecNum': ValVecNum}


    elif (inode == 4):

        npoly = len(IDReg)
        IDMag = discretizations['BS-1_Magnitude']['ID']
        IDMec = discretizations['BS-4_FocalMechanism']['ID']

        IDArea   = np.empty((npoly,len(IDMag),len(IDMec)),dtype=str)
        IDLength = np.empty((npoly,len(IDMag),len(IDMec)),dtype=str)
        ValArea  = np.empty((npoly,len(IDMag),len(IDMec)))
        ValLen   = np.empty((npoly,len(IDMag),len(IDMec)))

        for lines in f:
            IDRegTmp = lines[0:22]
            IDMagTmp = lines[23:27]   # 23-26
            IDMecTmp = lines[28:39]
            Val      = lines[41:]

            area     = float(Val.split()[0])
            length   = float(Val.split()[1])

            TmpPosReg = IDReg.index(IDRegTmp)
            TmpPosMag = IDMag.index(IDMagTmp)
            TmpPosMec = IDMec.index(IDMecTmp)


            IDArea[TmpPosReg,TmpPosMag,TmpPosMec]   = 'A' + str(int(area)).zfill(6)
            IDLength[TmpPosReg,TmpPosMag,TmpPosMec] = 'L' + str(int(length)).zfill(6)
            ValArea[TmpPosReg,TmpPosMag,TmpPosMec]  = area
            ValLen[TmpPosReg,TmpPosMag,TmpPosMec]   = length

        items = {'IDArea': IDArea, 'IDLength': IDLength, 'ValArea': ValArea, 'ValLen': ValLen}

    elif (inode == 5):

        npoly = len(IDReg)
        IDMag = discretizations['BS-1_Magnitude']['ID']
        IDMec = discretizations['BS-4_FocalMechanism']['ID']

        ID  = np.empty((npoly,len(IDMag),len(IDMec)),dtype=str)
        Val = np.empty((npoly,len(IDMag),len(IDMec)))

        for lines in f:
            IDRegTmp = lines[0:22]
            IDMagTmp = lines[23:27]   # 23-26
            IDMecTmp = lines[28:39]
            slip     = float(lines[41:])

            TmpPosReg = IDReg.index(IDRegTmp)
            TmpPosMag = IDMag.index(IDMagTmp)
            TmpPosMec = IDMec.index(IDMecTmp)

            ID[TmpPosReg,TmpPosMag,TmpPosMec] = 'S' + str(int(slip*10)).zfill(3)
            Val[TmpPosReg,TmpPosMag,TmpPosMec] = slip

        items = {'ID': ID, 'Val': Val}

    return items

def get_PS_discretizations(**kwargs):

    from shapely import geometry

    inode           = kwargs.get('i', None)
    #f = kwargs.get('txtfile', None)
    poly            = kwargs.get('regions_poly', None)
    bar_pos         = kwargs.get('bxy', None)
    bar_dep         = kwargs.get('bz', None)
    file_name       = kwargs.get('file_name', None)
    grid_bary       = kwargs.get('grid_bary', None)

    f = open(file_name, "r")

#0: PS-1_Magnitude
#1: PS-2_PositionArea

    if (inode == 0):
        ID = []
        Val = []

        for lines in f:
            foe = lines.rstrip().split(':')

            ID.append(foe[0].rstrip())
            Val.append(float(foe[1]))


        #Vale = np.array(Val)

        items = {'ID': ID,'Val': Val}

    elif (inode == 1):

        ID = []
        Val = []
        Val_x = []
        Val_y = []
        Region = []
        Depth = []

        coord_ref = []

        for lines in f:
            ID.append(lines[0:16])
            tmpcoords=lines[18:].split('[')[0]
            Val.append(tmpcoords)

            clon, clat = (float(x)  for x in tmpcoords.split())
            coord_ref.append([clon, clat])
            Val_x.append(clon)
            Val_y.append(clat)

            #print(clon, clat)

            nr=0
            for rp in poly:
                nr += 1
                if (rp.contains(geometry.Point(clon, clat))):
                    Region.append(nr)


        coord_ref = np.array(coord_ref)
        for i in range(len(coord_ref)):


            # cerca coppia di coordinate (grid_discr[i]) nell'arrey di coppie della maho
            c = np.where(grid_bary == coord_ref[i])
            #print(c)
            #sys.exit()
            # Per qualche ragione da un duplicato di indici all'indice che trova. Quindi trova indice doppio
            m = np.zeros_like(c[0], dtype=bool)
            m[np.unique(c[0], return_index=True)[1]] = True
            # Indice della moho che si cerca è questo:
            index=c[0][~m][0]
            #print(index)
            #print(bar_dep[index])
            #raccogli tutt ele profondità
            Depth.append(bar_dep[index])


        items = {'ID': ID,'Val': Val, 'Region': Region, 'Depth': Depth, 'Val_x' : Val_x, 'Val_y': Val_y}

    return items


def get_BS_models(**kwargs):

    inode = kwargs.get('i', None)
    f = kwargs.get('txtfile', None)

#0: BS-1_Magnitude
#1: BS-2_Position
#2: BS-3_Depth
#3: BS-4_FocalMechanism
#4: BS-5_Area
#5: BS-6_Slip

    if (inode == 0) or (inode == 3):

        Wei = []
        Type = []

        for lines in f:
            foe = lines.rstrip().split(':')

            lst = foe[1].rstrip().split()
            val = np.array([float(i) for i in lst])
            Wei.extend(val)
        nmodels=int(len(Wei)/2)
        Type = [1 for i in range(nmodels)]
        Type.extend([2 for i in range(nmodels,len(Wei))])

        items = {'Wei': Wei,'Type': Type}

    elif (inode == 1):

        Wei = []
        Type = []

        for lines in f:
            foe = lines.rstrip().split(':')
            Wei.append(float(foe[1]))
        nmodels=int(len(Wei)/2)
        Type = [1 for i in range(nmodels)]
        Type.extend([2 for i in range(nmodels,len(Wei))])

        items = {'Wei': Wei,'Type': Type}

    elif (inode == 2) or (inode == 4) or (inode == 5):

        Wei = [1]
        items = {'Wei': Wei}


    return items

def get_PS_models(**kwargs):

    f = kwargs.get('txtfile', None)

#0: PS-1_Magnitude
#1: PS-2_PositionArea

    Wei = []
    Type = []

    for lines in f:
        foe = lines.rstrip().split(':')

        lst = foe[1].rstrip().split()
        val = np.array([float(i) for i in lst])
        Wei.extend(val)
    nmodels=int(len(Wei)/2)
    Type = [1 for i in range(nmodels)]
    Type.extend([2 for i in range(nmodels,len(Wei))])

    items = {'Wei': Wei,'Type': Type}

    return items
