from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import os
import gdown
# 访问GoogleAPI的科学上网端口
import socks
import socket
socks.set_default_proxy(socks.PROXY_TYPE_HTTP, "127.0.0.1", 30008)
socket.socket = socks.socksocket

 
gauth = GoogleAuth()
gauth.settings['client_config_file'] = os.path.join(os.path.dirname(__file__), 'client_secret.json')
gauth.LocalWebserverAuth()
drive = GoogleDrive(gauth)
 
# 设置需要下载的Google Drive文件夹ID
parent_folder_id = '10CHMoCPLRuF_7ss9ZKuTLRX1QneeUd7k'
 
# 设置你在Bash上的下载路径。
parent_folder_dir = '.'
 
if parent_folder_dir[-1] != '/':
  parent_folder_dir = parent_folder_dir + '/'
 
# Get the folder structure
 
file_dict = dict()
folder_queue = [parent_folder_id]
dir_queue = [parent_folder_dir]
cnt = 0
 
while len(folder_queue) != 0:
    current_folder_id = folder_queue.pop(0)
    file_list = drive.ListFile({'q': "'{}' in parents and trashed=false".format(current_folder_id)}).GetList()

    current_parent = dir_queue.pop(0)
    print(current_parent, current_folder_id)
    for file1 in file_list:
        file_dict[cnt] = dict()
        file_dict[cnt]['id'] = file1['id']
        file_dict[cnt]['title'] = file1['title']
        file_dict[cnt]['dir'] = current_parent + file1['title']
        
        if file1['mimeType'] == 'application/vnd.google-apps.folder':
            file_dict[cnt]['type'] = 'folder'
            file_dict[cnt]['dir'] += '/'
            folder_queue.append(file1['id'])
            dir_queue.append(file_dict[cnt]['dir'])
            os.makedirs(current_parent + file1['title'], exist_ok=True)
        else:
            file_dict[cnt]['type'] = 'file'
            file_dict[cnt]['md5Checksum'] = file1['md5Checksum']
            gdown.download("https://drive.google.com/uc?id=%s"%file1['id'], current_parent + file1['title'], quiet=False)
        cnt += 1
 