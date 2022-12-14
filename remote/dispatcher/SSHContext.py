#!/anaconda3/envs/schnet/bin/python
# -*- coding:utf-8 -*-
# author: Tonghe Ying

import os, sys, paramiko, json, uuid, tarfile, time, stat, shutil
from glob import glob


class SSHSession(object):
    def __init__(self, jdata):
        self.remote_profile = jdata
        self.remote_host = '211.86.151.104'
        self.remote_uname = 'yingth'
        self.remote_port = 22
        self.remote_password = None
        self.local_key_filename = None
        self.remote_timeout = None
        self.local_key_passphrase = None
        self.remote_workpath = jdata['remote_workpath']
        self.ssh = None
        self._setup_ssh(hostname=self.remote_host,
                        port=self.remote_port,
                        username=self.remote_uname,
                        password=self.remote_password,
                        key_filename=self.local_key_filename,
                        timeout=self.remote_timeout,
                        passphrase=self.local_key_passphrase)

    def _setup_ssh(self,
                   hostname,
                   port=22,
                   username=None,
                   password=None,
                   key_filename=None,
                   timeout=None,
                   passphrase=None):
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        pkey = paramiko.RSAKey.from_private_key_file('/home/yingth/.ssh/id_rsa_ustc_pso')
        self.ssh.connect(hostname='211.86.151.104', port=22, username='yingth', pkey=pkey)
        assert(self.ssh.get_transport().is_active())
        transport = self.ssh.get_transport()
        transport.set_keepalive(60)

    def ensure_alive(self,
                     max_check=10,
                     sleep_time=10):
        count = 1
        while not self._check_alive():
            if count == max_check:
                raise RuntimeError('cannot connect ssh after %d failures at interval %d s' %
                                   (max_check, sleep_time))
            self._setup_ssh(hostname=self.remote_host,
                            port=self.remote_port,
                            username=self.remote_uname,
                            password=self.remote_password,
                            key_filename=self.local_key_filename,
                            timeout=self.remote_timeout,
                            passphrase=self.local_key_passphrase)
            count += 1
            time.sleep(sleep_time)

    def _check_alive(self):
        if self.ssh is None:
            return False
        try:
            transport = self.ssh.get_transport()
            transport.send_ignore()
            return True
        except EOFError:
            return False

    def get_session_root(self):
        return self.remote_workpath

    def get_ssh_client(self):
        return self.ssh


class SSHContext(object):
    def __init__(self,
                 local_root='/Users/tongheying/automation/test',
                 ssh_session=None,
                 job_uuid=None):
        assert (type(local_root) == str)
        self.local_root = os.path.abspath(local_root)
        if job_uuid:
            self.job_uuid = job_uuid
        else:
            self.job_uuid = str(uuid.uuid4())  # the id code
            # self.job_uuid = 'ssh_001'
        self.remote_root = os.path.join(ssh_session.get_session_root(), self.job_uuid)
        self.ssh_session = ssh_session
        self.ssh_session.ensure_alive()
        try:
            sftp = self.ssh_session.ssh.open_sftp()
            sftp.mkdir(self.remote_root)
            sftp.close()
        except:
            pass

    @property
    def ssh(self):
        return self.ssh_session.get_ssh_client()

    def upload(self,
               job_dirs,
               local_up_files,
               dereference=True):
        self.ssh_session.ensure_alive()
        cwd = os.getcwd()
        os.chdir(self.local_root)
        file_list = []
        for ii in job_dirs:
            for jj in local_up_files:
                file_list.append(os.path.join(ii, jj))
        self._put_files(file_list, dereference=dereference)
        os.chdir(cwd)

    def download(self,
                 job_dirs,
                 remote_down_files,
                 check_exist=False,
                 mark_failure=True,
                 back_error=False):
        self.ssh_session.ensure_alive()
        cwd = os.getcwd()
        os.chdir(self.local_root)
        file_list = []
        for ii in job_dirs:
            for jj in remote_down_files:
                file_name = os.path.join(ii, jj)
                if check_exist:
                    if self.check_file_exists(file_name):
                        file_list.append(file_name)
                    elif mark_failure:
                        with open(os.path.join(self.local_root, ii, 'tag_failure_download_%s' % jj), 'w') as fp: pass
                    else:
                        pass
                else:
                    file_list.append(file_name)
            if back_error:
                errors = glob(os.path.join(ii, 'error*'))
                file_list.extend(errors)
        if len(file_list) > 0:
            self._get_files(file_list)
        os.chdir(cwd)

    def read_file(self, fname):
        self.ssh_session.ensure_alive()
        sftp = self.ssh.open_sftp()
        with sftp.open(os.path.join(self.remote_root, fname), 'r') as fp:
            ret = fp.read().decode('utf-8')
        sftp.close()
        return ret

    def check_file_exists(self, fname):
        self.ssh_session.ensure_alive()
        sftp = self.ssh.open_sftp()
        try:
            sftp.stat(os.path.join(self.remote_root, fname))  # get file information
            ret = True
        except IOError:
            ret = False
        sftp.close()
        return ret

    def block_checkcall(self,
                        cmd):
        self.ssh_session.ensure_alive()
        stdin, stdout, stderr = self.ssh.exec_command(('cd %s ;' % self.remote_root) + cmd)
        exit_status = stdout.channel.recv_exit_status()  # check whether jobs is done normally
        if exit_status != 0:
            raise RuntimeError("Get error code %d in calling %s through ssh with job: %s . message: %s" %
                               (exit_status, cmd, self.job_uuid, stderr.read().decode('utf-8')))
        return stdin, stdout, stderr

    # run command
    def block_call(self,
                   cmd):
        self.ssh_session.ensure_alive()
        stdin, stdout, stderr = self.ssh.exec_command(('cd %s ;' % self.remote_root) + cmd)
        exit_status = stdout.channel.recv_exit_status()
        return exit_status, stdin, stdout, stderr

    def clean(self):
        self.ssh_session.ensure_alive()
        sftp = self.ssh.open_sftp()
        self._rmtree(sftp, self.remote_root)
        sftp.close()

    def write_file(self, fname, write_str):  # write files
        self.ssh_session.ensure_alive()
        sftp = self.ssh.open_sftp()
        with sftp.open(os.path.join(self.remote_root, fname), 'w') as fp:
            fp.write(write_str)
        sftp.close()

    def _rmtree(self, sftp, remotepath, level=0, verbose=False):
        for f in sftp.listdir_attr(remotepath):
            rpath = os.path.join(remotepath, f.filename)
            if stat.S_ISDIR(f.st_mode):
                self._rmtree(sftp, rpath, level=(level + 1))
            else:
                rpath = os.path.join(remotepath, f.filename)
                if verbose: print('removing %s%s' % ('    ' * level, rpath))
                sftp.remove(rpath)
        if verbose: print('removing %s%s' % ('    ' * level, remotepath))
        sftp.rmdir(remotepath)

    def _put_files(self,
                   files,
                   dereference=True):
        of = self.job_uuid + '.tgz'
        # local tar
        cwd = os.getcwd()
        os.chdir(self.local_root)
        if os.path.isfile(of):
            os.remove(of)
        with tarfile.open(of, "w:gz", dereference=dereference) as tar:
            for ii in files:
                tar.add(ii)
        os.chdir(cwd)
        # trans
        from_f = os.path.join(self.local_root, of)
        to_f = os.path.join(self.remote_root, of)
        sftp = self.ssh.open_sftp()
        try:
            sftp.put(from_f, to_f)
        except FileNotFoundError:
            raise FileNotFoundError("from %s to %s Error!" % (from_f, to_f))
        # remote extract
        self.block_call('tar xf %s' % of)
        # clean up
        os.remove(from_f)
        sftp.remove(to_f)
        sftp.close()

    def _get_files(self,
                   files):
        of = self.job_uuid + '.tgz'
        flist = ""
        for ii in files:
            flist += " " + ii
        # remote tar
        self.block_checkcall('tar czf %s %s' % (of, flist))
        # trans
        from_f = os.path.join(self.remote_root, of)
        to_f = os.path.join(self.local_root, of)
        if os.path.isfile(to_f):
            os.remove(to_f)
        sftp = self.ssh.open_sftp()
        sftp.get(from_f, to_f)
        # extract
        cwd = os.getcwd()
        os.chdir(self.local_root)
        with tarfile.open(of, "r:gz") as tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar)
        os.chdir(cwd)
        # cleanup
        os.remove(to_f)
        sftp.remove(from_f)
        sftp.close()
