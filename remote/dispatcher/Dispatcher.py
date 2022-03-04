#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: Tonghe Ying

import os, sys, time, random, json, glob
from hashlib import sha1
from remote.dispatcher.SSHContext import SSHSession, SSHContext
from remote.dispatcher.LSF import LSF
from remote.dispatcher.JobStatus import JobStatus


def _split_tasks(tasks,
                 group_size):
    ntasks = len(tasks)
    ngroups = ntasks // group_size
    if ngroups * group_size < ntasks:
        ngroups += 1
    chunks = [[]] * ngroups
    tot = 0
    for ii in range(ngroups):
        chunks[ii] = (tasks[ii::ngroups])
        tot += len(chunks[ii])
    assert (tot == len(tasks))
    return chunks


class Dispatcher(object):
    def __init__(self, remote_profile,
                 context_type='ssh',
                 batch_type='lsf',
                 job_record='jr.json'):
        self.remote_profile = remote_profile

        self.session = SSHSession(remote_profile)
        self.context = SSHContext
        self.uuid_name = True

        self.batch = LSF

        self.jrname = job_record

    def run_jobs(self,
                 resources,
                 command,
                 work_path,
                 tasks,
                 group_size,
                 forward_common_files,
                 forward_task_files,
                 backward_task_files,
                 forward_task_deference=True,
                 mark_failure=False,
                 outlog='log',
                 errlog='err',
                 args=None,
                 natoms=None):
        job_handler = self.submit_jobs(resources,
                                       command,
                                       work_path,
                                       tasks,
                                       group_size,
                                       forward_common_files,
                                       forward_task_files,
                                       backward_task_files,
                                       forward_task_deference,
                                       outlog,
                                       errlog,
                                       args,
                                       natoms=natoms)
        while not self.all_finished(job_handler, mark_failure):
            time.sleep(60)  # check every 5 minutes

    def submit_jobs(self,
                    resources,
                    command,
                    work_path,
                    tasks,
                    group_size,
                    forward_common_files,
                    forward_task_files,
                    backward_task_files,
                    forward_task_deference=True,
                    outlog='log',
                    errlog='err',
                    args=None,
                    natoms=None):
        self.backward_task_files = backward_task_files
        task_chunks = _split_tasks(tasks, group_size)
        task_chunks_str = ['+'.join(ii) for ii in task_chunks]
        task_hashes = [sha1(ii.encode('utf-8')).hexdigest() for ii in task_chunks_str]
        job_record = JobRecord(work_path, task_chunks, fname=self.jrname)
        job_record.dump()
        nchunks = len(task_chunks)

        job_list = []
        for ii in range(nchunks):
            cur_chunk = task_chunks[ii]
            cur_hash = task_hashes[ii]
            if not job_record.check_finished(cur_hash):
                submitted = job_record.check_submitted(cur_hash)
                if not submitted:
                    job_uuid = None
                else:
                    job_uuid = job_record.get_uuid(cur_hash)
                # communication context, batch system
                context = self.context(work_path, self.session, job_uuid)  # every task_chunk has an independent job_uuid
                batch = self.batch(context, uuid_names=self.uuid_name)
                rjob = {'context': context, 'batch': batch}
                # upload files
                if not rjob['context'].check_file_exists(rjob['batch'].upload_tag_name):
                    if forward_common_files[0] != '':
                        rjob['context'].upload('.',
                                               forward_common_files)  # forward_common_files are files to be uploaded

                    rjob['context'].upload(cur_chunk,
                                           forward_task_files[cur_chunk[0]],
                                           dereference=forward_task_deference)
                    if nchunks == 4 and ii == 0:
                        tot = int(cur_chunk[0][-5:-2])
                        num = len(forward_task_files[cur_chunk[0]])
                        for jj in range(tot, tot-num, -1):
                            jj_str = str(jj).zfill(3)
                            rjob['context'].block_call('cd ' + cur_chunk[0] + ' && rm -rf '
                                                                              '../../dataset-ML/N'+str(natoms)+'/N'+str(natoms)+'_dataset/updating_' +
                                                       jj_str + ' && cp -r ../train_'+jj_str+'-1/updating_' + jj_str +
                                                       ' ../../dataset-ML/N'+str(natoms)+'/N'+str(natoms)+'_dataset/')
                    rjob['context'].write_file(rjob['batch'].upload_tag_name, '')
                # submit new or recover old submission
                if not submitted:
                    rjob['batch'].submit(cur_chunk, command, args=args, res=resources, outlog=outlog, errlog=errlog)
                    job_uuid = rjob['context'].job_uuid
                else:
                    rjob['batch'].submit(cur_chunk, command, args=args, res=resources,
                                         outlog=outlog, errlog=errlog, restart=True)
                # record job and its remote context
                job_list.append(rjob)
                ip = None
                instance_id = None
                job_record.record_remote_context(cur_hash,
                                                 context.local_root,
                                                 context.remote_root,
                                                 job_uuid,
                                                 ip,
                                                 instance_id)
                job_record.dump()
            else:
                job_list.append(None)
        assert (len(job_list) == nchunks)
        job_handler = {
            'task_chunks': task_chunks,
            'job_list': job_list,
            'job_record': job_record,
            'command': command,
            'resources': resources,
            'outlog': outlog,
            'errlog': errlog,
            'backward_task_files': backward_task_files,
            'args': args
        }
        return job_handler

    def all_finished(self,
                     job_handler,
                     mark_failure,
                     clean=True):
        task_chunks = job_handler['task_chunks']
        task_chunks_str = ['+'.join(ii) for ii in task_chunks]
        task_hashes = [sha1(ii.encode('utf-8')).hexdigest() for ii in task_chunks_str]  # the same string corresponds the same number
        job_list = job_handler['job_list']
        job_record = job_handler['job_record']
        command = job_handler['command']
        tag_failure_list = ['tag_failure_%d' % ii for ii in range(len(command))]
        resources = job_handler['resources']
        outlog = job_handler['outlog']
        errlog = job_handler['errlog']
        backward_task_files = job_handler['backward_task_files']
        args = job_handler['args']
        print('checking jobs')
        nchunks = len(task_chunks)
        for idx in range(nchunks):
            cur_hash = task_hashes[idx]
            rjob = job_list[idx]
            if not job_record.check_finished(cur_hash):
                # chunk not finished according to record
                status = rjob['batch'].check_status()
                job_uuid = rjob['context'].job_uuid
                print('checked job %s' % job_uuid)
                if status == JobStatus.terminated:
                    job_record.increase_nfail(cur_hash)
                    if job_record.check_nfail(cur_hash) > 3:
                        raise RuntimeError('Job %s failed for more than 3 times' % job_uuid)
                    print('job %s terminated, submit again' % job_uuid)
                    print('try %s times for %s' % (job_record.check_nfail(cur_hash), job_uuid))
                    rjob['batch'].submit(task_chunks[idx], command, args=args, res=resources, outlog=outlog,
                                         errlog=errlog, restart=True)
                elif status == JobStatus.finished:
                    print('job %s finished' % job_uuid)
                    if mark_failure:
                        rjob['context'].download(task_chunks[idx], tag_failure_list, check_exists=True,
                                                 mark_failure=False)
                        rjob['context'].download(task_chunks[idx], backward_task_files[task_chunks[idx]],
                                                 check_exists=True)
                    else:
                        if task_chunks[idx][0].startswith('correction'):
                            rjob['context'].block_call('cd ' + task_chunks[idx][0] + ' && python read_static.py')
                        rjob['context'].download(task_chunks[idx], backward_task_files[task_chunks[idx][0]])
                    if clean:
                        rjob['context'].clean()
                    job_record.record_finish(cur_hash)
                    job_record.dump()
        job_record.dump()
        return job_record.check_all_finished()


class JobRecord(object):
    def __init__(self, path, task_chunks, fname='job_record.json', ip=None):
        self.path = os.path.abspath(path)
        self.fname = os.path.join(self.path, fname)
        self.task_chunks = task_chunks
        if not os.path.exists(self.fname):
            self._new_record()
        else:
            self.load()

    def check_submitted(self, chunk_hash):
        self.valid_hash(chunk_hash)
        return self.record[chunk_hash]['context'] is not None

    def record_remote_context(self,
                              chunk_hash,
                              local_root,
                              remote_root,
                              job_uuid,
                              ip=None,
                              instance_id=None):
        self.valid_hash(chunk_hash)
        self.record[chunk_hash]['context'] = {}
        self.record[chunk_hash]['context']['local_root'] = local_root
        self.record[chunk_hash]['context']['remote_root'] = remote_root
        self.record[chunk_hash]['context']['job_uuid'] = job_uuid
        self.record[chunk_hash]['context']['ip'] = ip
        self.record[chunk_hash]['context']['instance_id'] = instance_id

    def get_uuid(self, chunk_hash):
        self.valid_hash(chunk_hash)
        return self.record[chunk_hash]['context']['job_uuid']

    def check_finished(self, chunk_hash):
        self.valid_hash(chunk_hash)
        return self.record[chunk_hash]['finished']

    def check_all_finished(self):
        flist = [self.record[ii]['finished'] for ii in self.record]
        return all(flist)

    def record_finish(self, chunk_hash):
        self.valid_hash(chunk_hash)
        self.record[chunk_hash]['finished'] = True

    def check_nfail(self, chunk_hash):
        self.valid_hash(chunk_hash)
        return self.record[chunk_hash]['fail_count']

    def increase_nfail(self, chunk_hash):
        self.valid_hash(chunk_hash)
        self.record[chunk_hash]['fail_count'] += 1

    def valid_hash(self, chunk_hash):
        if chunk_hash not in self.record.keys():
            raise RuntimeError('chunk hash %s not in record, an invalid record may be used, please check file %s' % (
            chunk_hash, self.fname))

    def _new_record(self):
        task_chunks_str = ['+'.join(ii) for ii in self.task_chunks]
        task_hash = [sha1(ii.encode('utf-8')).hexdigest() for ii in task_chunks_str]
        self.record = {}
        for ii, jj in zip(task_hash, self.task_chunks):
            self.record[ii] = {
                'context': None,
                'finished': False,
                'fail_count': 0,
                'task_chunk': jj
            }

    def dump(self):
        with open(self.fname, 'w') as fp:
            json.dump(self.record, fp, indent=4)  # write data into json

    def load(self):
        with open(self.fname) as fp:
            self.record = json.load(fp)  # read from json and store with dictionary type


def make_dispatcher(jdata=None):
    context_type = 'ssh'
    batch_type = 'lsf'
    disp = Dispatcher(jdata, context_type=context_type, batch_type=batch_type)
    return disp
