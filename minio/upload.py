from minio import Minio
import os
import queue
import threading
import random
import argparse
import tqdm
import time


import glob

class UploadPool:

    def __init__(self, client, local_path, bucket_name, minio_path, t=10) -> None:
        self.queues = [ queue.Queue() for _ in range(t) ]
        self.errors = [] # list for errors from threads.
        self.threads = [ threading.Thread(target=self.run, args=(i,)) for i in range(t) ] # thread pool for minio
        self.client = client
        self.sucess_count = 0
        self.error_count  = 0
        self.s = 0
        self._pbar = tqdm.tqdm(total=0)
        self._isend = False
        self._total = 0
        self._find_end = False
        self.path = [local_path, bucket_name, minio_path]

    
        


    def run(self, q=0) -> None: # thread function for each thread.  takes items from queue and puts them in minio server.  then puts
        queue = self.queues[q]
        print('%d start' % q)
        while True:
            item = queue.get()
            try: # try catch for minio client.  if it fails, put it back on the queue for another thread to take it from.  if it
                bucket_name, remote_path, local_file = item
                self.client.fput_object(bucket_name,  # put object into minio server.  bucket name is the same as the
                                        remote_path,
                                        local_file)
                queue.task_done() # put it in the queue.  if it works, it will be put back on the queue for another thread to take it
                self.sucess_count += 1
                # print('%d upload'%q)
            except Exception as e:
                self.errors.append([item, e]) # if it fails, put it back on the queue for another thread to take it from.  if it is a
                self.error_count += 1
                # print(e)

            
    def upload(self, bucket_name, remote_path, local_file):
        self.s += 1
        self.s %= len(self.t)
        self.queues[self.s].put((bucket_name, remote_path, local_file), block=False)
        # random.choice(self.queues) # if all threads are busy, put the item in the

    def find(self):
        local_path, bucket_name, minio_path = self.path
        self._find_end = False
        if isinstance(local_path, str):
            self._find(local_path, bucket_name, minio_path)
        elif isinstance(local_path, (list, tuple)):
            for p in local_path: # if it is a list or tuple, then it is a pattern to search for.  for each pattern, find all files that match
                self._find(p, bucket_name, minio_path)
    
        self._find_end = True

    def start(self):
        self.find()
        for t in self.threads: t.start() # start threads.
        for t in self.threads: t.join() # wait for threads to finish.
        bar_thread = threading.Thread(target=self.pbar) # create a thread to print the progress bar.
        bar_thread.start()
        bar_thread.join()

        while True:
            if not self._find_end:
                continue
            
            isend = self._total <= (self.sucess_count + self.error_count) # if all files are successfully upload, break the loop.  if all files fail, break

            if isend:
                break
        
        self._isend = True

        for t in self.threads:
            del t

        with open("log-%d.txt"%(time.time() * 1000), "w") as logfile: 
            if len(self.errors) > 0:
                for item, e in self.errors:
                    logfile.write(f"{item}\t{e}\n")
                logfile.flush() # flush the logfile buffer.  this is important.  if there are large files, this can cause a memory




    def _find(self, local_path, bucket_name, minio_path):
        assert os.path.isdir(local_path)

        for local_file in glob.glob(local_path + '/**'):
            # print(local_file)
            local_file = local_file.replace(os.sep, "/") # Replace \ with / on Windows
            if not os.path.isfile(local_file):
                self._find(
                    local_file, bucket_name, minio_path + "/" + os.path.basename(local_file))
            else:
                remote_path = os.path.join(
                    minio_path, local_file[1 + len(local_path):])
                remote_path = remote_path.replace(
                    os.sep, "/")  # Replace \ with / on Windows
                self.upload(bucket_name, remote_path, local_file)
                self._total += 1
                
    def pbar(self):
        while not self._isend:
            time.sleep(1)
            self._pbar.set_postfix(
                dict(
                    succ=self.sucess_count, err=self.error_count, tot=self._total
                )
            )
        

if __name__ == '__main___':

    parser = argparse.ArgumentParser(
        'MinIO 文件上传助手，用于文件夹上传，使用多线程方式上传。该脚本在查找到文件后会立即上传，不会进行时间统计，也无法统计准确时间。上传失败的文件不会重试，但可以保存错误文件列表', 
    )

    parser.add_argument(
        '--endpoint', '-e', type=str, required=True, help='MinIO 服务器地址',
    )
    parser.add_argument(
        '--secure', default=False, action='store_true', help='Use HTTPS instead of HTTP, default is HTTP'
    )

    parser.add_argument(
        '--bucket', '-b', type=str, required=True, help='Bucket name'
    )

    parser.add_argument(
        '--accesskey', '-a', type=str, required=True, help='Access key'
    )

    parser.add_argument(
        '--secretkey', '-s', type=str, required=True, help='Secret key'
    )

    parser.add_argument(
        '--threads', '-t', type=int, default=5, help='Number of threads, default is 5'
    )

    parser.add_argument(
        '--dir', '-d', type=str, default='.', help='Directory to upload'
    )

    parser.add_argument(
        '--remote-path', '-r', type=str, default='', help='Remote path '  #
    )

    args = parser.parse_args()  # Get arguments from command line as parsed by argparse.ArgumentParser() class. 创建参数对象
    endpoint = args.endpoint  # Set endpoint as parsed argument. 保存参数对象到变量 endpont. 在


    client = Minio(
        endpoint=args.endpoint,
        secure=args.secure,
        access_key=args.accesskey,
        secret_key=args.secretkey
    )
    try:
        if not client.bucket_exists(args.bucket):
            client.make_bucket(args.bucket)
    
        tool = UploadPool(client, args.dir, args.bucket, args.remote_path, args.threads)
        tool.start()
        
    except Exception as e: # This exception handling should be generic and catch all exceptions. 如果出现某些错误，则应显示错误
        print(e)
    