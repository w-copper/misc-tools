from .rotutils import get_matrix
import xmltodict
import os

def parser_ccxml(ccxml_path, metadata_path):
    '''
    TODO: metadata 不在使用,去除相关代码;
    TODO: 视场角仍未能通过参数确定,需要进一步设置
    '''
    # SRS = ''
    # ORIGIN = []
    # with open(metadata_path, 'r') as f:
    #     for line in f.readlines():
    #         if line.strip().startswith('<SRS>'):
    #             SRS = CRS.from_string(line.strip()[len('<SRS>'):-len('</SRS>')])
    #         if line.strip().startswith('<SRSOrigin>'):
    #             origin_str = line.strip()[len('<SRSOrigin>'):-len('</SRSOrigin>')]
    #             origin_ls = origin_str.split(',')
    #             ORIGIN.extend((float(origin_ls[0]), float(origin_ls[1]), float(origin_ls[2])))
    
    # ORIGIN = np.array(ORIGIN).reshape(3)
    # BLOKC_SRSS = dict()

    # ZINV = np.eye(4)
    # ZINV[2,2] = -1
    # R90 = get_matrix(0, 0, -90, np.zeros(3)) # look at y, z up x right

    # return

    def read_to_line(f, endline = None):
        if endline is None:
            return f.readlines()
        else:
            res = []
            while True:
                line = f.readline()
                if line.strip() == endline:
                    return res + [line]
                else:
                    res.append(line)
    
    with open(ccxml_path, 'r') as f:
        line = f.readline().strip()
        # first_line = line
        # current_block_srs = None
        current_image_size = [0, 0]
        current_aspec = 1
        current_distortion = [ 0.0, 0.0, 0.0, 0.0, 0 ]
        current_focallength = 0
        while line != '</BlocksExchange>':
            if line.startswith('<SpatialReferenceSystems>'):
                # logging.info('process srs')
                
                res = read_to_line(f, '</SpatialReferenceSystems>')       
                line = f.readline().strip()         
                continue
                res.insert(0, line)
                xmlstr = 	'\n'.join(res)

                srss = xmltodict.parse(xmlstr)
                for srs in srss['SpatialReferenceSystems']['SRS']:
                    BLOKC_SRSS[srs['Id']] = CRS.from_string(srs['Definition'])
                # logging.info('fine %d srs', len(BLOKC_SRSS))
            elif line.startswith('<SRSId>'):
                line = f.readline().strip()
                continue
                current_block_srs = BLOKC_SRSS[line.split('>')[1].split('<')[0]]
            elif line.startswith('<ImageDimensions>'):
                wline = f.readline().strip().split('>')[1].split('<')[0]
                hline = f.readline().strip().split('>')[1].split('<')[0]
                current_image_size[0] = int(wline)
                current_image_size[1] = int(hline)
            elif line.startswith('<AspectRatio>'):
                current_aspec = float(f.readline().strip().split('>')[1].split('<')[0])
            elif line.startswith('<FocalLength>'):
                current_focallength = 	float(line.split('>')[1].split('<')[0])
            elif line.startswith('<Photo>'):
                res = read_to_line(f, '</Photo>')
                res.insert(0, line)
                xmlstr = 	'\n'.join(res)
                image_info = xmltodict.parse(xmlstr)['Photo']
                image_pose = image_info['Pose']
                yaw = float(image_pose['Rotation']['Yaw'])
                pitch = float(image_pose['Rotation']['Pitch'])
                roll =  float(image_pose['Rotation']['Roll'])

                R90 = get_matrix(0, -90, 0, np.zeros(3)) # look at y, z up x right
                YAW = get_matrix(0, 0, yaw, np.zeros(3)) # cz
                PITCH = get_matrix(0, -pitch, 0, np.zeros(3)) # ccx
                ROLL = get_matrix(-roll, 0, 0, np.zeros(3)) # cy
                # T = get_matrix(0, 0, 0, np.array([camera_pos[-1]])) # T

                pymatrix = ROLL @ PITCH @ YAW @ R90
                
                # yaw = 180 + yaw
                # if yaw < 0:
                #     yaw = yaw + 360
                # roll, pitch = -pitch, roll
                # rot_matrix = [[ float(image_pose['Rotation']['M_00']), float(image_pose['Rotation']['M_01']), float(image_pose['Rotation']['M_02'])],
                #              [float(image_pose['Rotation']['M_10']), float(image_pose['Rotation']['M_11']), float(image_pose['Rotation']['M_12'])],           
                #              [float(image_pose['Rotation']['M_20']), float(image_pose['Rotation']['M_21']), float(image_pose['Rotation']['M_22'])]]
                position = [float(image_pose['Center']['x']), 	float(image_pose['Center']['y']), 	float(image_pose['Center']['z'])]
                # position = pyproj.transform(current_block_srs, SRS, *position)
                # position = Transformer.from_crs(current_block_srs, SRS).transform(*position)
                # position = np.array(position).reshape(3)
                # relposition = position[[1,0,2]]   - ORIGIN
                relposition = position
                pymatrix[:3,3] = relposition[:]
                img_name = os.path.basename(image_info['ImagePath'])
                img_name = os.path.splitext(img_name)[0]
                camrea = dict(
                    yfov= 40 / 180 * np.pi,
                    aspec = current_image_size[0] / current_image_size[1],
                    width = 	current_image_size[0],
                    height = 	current_image_size[1],
                    postype = 'matrix',
                    matrix = pymatrix,
                    yaw = yaw,
                    pitch = pitch,
                    roll = roll,
                    pos = relposition,
                    fname = img_name
                )
                # print(camrea)
                yield camrea
            
            line = f.readline().strip()
    