#!/anaconda3/envs/schnet/bin/python
# -*- coding:utf-8 -*-
# author: Tonghe Ying

import os,sys,time
from remote.dispatcher.JobStatus import JobStatus


class Batch(object):
    def __init__(self,
                 context=None,
                 uuid_names=True):
        self.context = context
        self.uuid_names = uuid_names
        if uuid_names:
            self.upload_tag_name = '%s_tag_upload' % self.context.job_uuid
            self.finish_tag_name = '%s_tag_finished' % self.context.job_uuid
            self.sub_script_name = '%s.sub' % self.context.job_uuid
            self.job_id_name = '%s_job_id' % self.context.job_uuid
        else:
            self.upload_tag_name = 'tag_upload'
            self.finish_tag_name = 'tag_finished'
            self.sub_script_name = 'run.sub'
            self.job_id_name = 'job_id'

    def check_status(self):
        raise RuntimeError('abstract method check_status should be implemented by derived class')

    def default_resources(self, res):
        raise RuntimeError('abstract method sub_script_head should be implemented by derived class')

    def sub_script_head(self, res):
        raise RuntimeError('abstract method sub_script_head should be implemented by derived class')

    def sub_script_cmd(self, cmd, arg, res):
        raise RuntimeError('abstract method sub_script_cmd should be implemented by derived class')

    def do_submit(self,
                  job_dirs,
                  cmd,
                  args=None,
                  res=None,
                  outlog='log',
                  errlog='err'):
        """
        submit a single job, assuming that no job is running there.
        """
        raise RuntimeError('abstract method do_submit should be implemented by derived class')

    def check_finish_tag(self):
        return self.context.check_file_exists(self.finish_tag_name)

    def sub_script(self,
                   job_dirs,
                   cmd,
                   args=None,
                   res=None,
                   outlog='log',
                   errlog='err'):
        """
        make submit script
        job_dirs(list):         directories of jobs. size: n_job
        cmd(list):              commands to be executed. size: n_cmd
        args(list of list):     args of commands. size of n_cmd x n_job
                                can be None
        res(dict):              resources available
        outlog(str):            file name for output
        errlog(str):            file name for error
        """
        res = self.default_resources(res)
        ret = self.sub_script_head(res)  # write head file
        if not isinstance(cmd, list):
            cmd = [cmd]
        if args is None:
            args = []
            for ii in cmd:
                _args = []
                for jj in job_dirs:
                    _args.append('')
                args.append(_args)

        self.cmd_cnt = 0
        try:
            self.manual_cuda_devices = res['manual_cuda_devices']
        except KeyError:
            self.manual_cuda_devices = 0
        try:
            self.manual_cuda_multiplicity = res['manual_cuda_multiplicity']
        except KeyError:
            self.manual_cuda_multiplicity = 1
        for ii in range(len(cmd)):
            ret += self._sub_script_inner(job_dirs,
                                          cmd[ii],
                                          args,
                                          ii,
                                          res,
                                          outlog=outlog,
                                          errlog=errlog)
        ret += '\ntouch %s\n' % self.finish_tag_name
        return ret

    def submit(self,
               job_dirs,
               cmd,
               args=None,
               res=None,
               restart=False,
               outlog='log',
               errlog='err'):
        if restart:
            print('restart task')
            status = self.check_status()
            if status in [JobStatus.unsubmitted, JobStatus.unknown, JobStatus.terminated]:
                self.do_submit(job_dirs, cmd, args, res, outlog=outlog, errlog=errlog)
            elif status == JobStatus.waiting:
                print('task is waiting')
            elif status == JobStatus.running:
                print('task is running')
            elif status == JobStatus.finished:
                print('task is finished')
            else:
                raise RuntimeError('unknown job status, must be wrong')
        else:
            print('new task')
            self.do_submit(job_dirs, cmd, args, res, outlog=outlog, errlog=errlog)
        if res is None:
            sleep = 0
        else:
            sleep = res.get('submit_wait_time', 0)  # res is a dictinary, storing information about machine
        time.sleep(sleep)

    def _sub_script_inner(self,
                          job_dirs,
                          cmd,
                          args,
                          idx,
                          res,
                          outlog='log',
                          errlog='err'):
        ret = ""
        allow_failure = res.get('allow_failure', False)
        for ii, jj in zip(job_dirs, args):
            ret += '#BSUB -q %s\n' % args[ii][2]
            ret += '#BSUB -n %s\n' % args[ii][3]
            ret += '\n\n'
            ret += 'cd %s\n' % ii
            ret += 'test $? -ne 0 && exit 1\n\n'  # test whether last command has been run correctly
            if self.manual_cuda_devices <= 0:
                ret += 'if [ ! -f tag_%d_finished ] ;then\n' % idx
                if args['add_sub']:
                    ret += '  rm -f id_prop.csv\n'
                    ret += '  for ((a = 0; a < '+args[ii][0]+'; a++))\n'
                    ret += '    do\n'
                    ret += '      c=$(printf "%04d" "$a")\n'
                    ret += '      echo "c=$c"\n'
                    ret += '      mkdir $c\n'
                    ret += '      cp INCAR $c\n'
                    ret += '      cp KPOINTS $c\n'
                    ret += '      cp POTCAR $c\n'
                    ret += '      cp '+args[ii][1]+'/POSCAR_$c $c/POSCAR\n'
                    ret += '      cd $c\n'
                    ret += '\n'
                    ret += '  %s 1>> %s 2>> %s \n' % (self.sub_script_cmd(cmd, jj, res), outlog, errlog)
                else:
                    ret += "  echo 'Start at:' $(date '+%y/%m/%d %H:%M:%S') > stdout\n"
                    ret += "  python -u ../../train_cluster.py " \
                           "-gen "+args[ii][0]+" -n "+ii[-1]+" -data "+args[ii][1]+" -natom "+args[ii][4] + " 1>> %s 2>> %s \n" % (outlog, errlog)
                if args['add_sub']:
                    ret += '\n'
                    ret += '      cd ..\n'
                    ret += '    done\n'
                else:
                    ret += "  echo 'Stop at:' $(date '+%y/%m/%d %H:%M:%S') >> stdout\n"
                if res['allow_failure'] is False:
                    ret += '  if test $? -ne 0; then exit 1; else touch tag_%d_finished; fi \n' % idx
                else:
                    ret += '  if test $? -ne 0; then touch tag_failure_%d; fi \n' % idx
                    ret += '  touch tag_%d_finished \n' % idx
                ret += 'fi\n\n'
            else:
                # do not support task-wise restart
                tmp_cmd = ' %s 1>> %s 2>> %s ' % (self.sub_script_cmd(cmd, jj, res), outlog, errlog)
                ret += 'export CUDA_VISIBLE_DEVICES=%d; %s &\n\n' % ((self.cmd_cnt % self.manual_cuda_devices), tmp_cmd)
                self.cmd_cnt += 1
            ret += 'cd %s\n' % self.context.remote_root
            ret += 'test $? -ne 0 && exit 1\n'
            if self.manual_cuda_devices > 0 and self.cmd_cnt % (
                    self.manual_cuda_devices * self.manual_cuda_multiplicity) == 0:
                ret += '\nwait\n\n'
        ret += '\nwait\n\n'  # wait all subprocess
        return ret



