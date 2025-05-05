# #! /usr/bin/env python
# #encoding:utf-8
#
# """
# python version：2.7
# Created on Oct 10  2016
# @author: liujm
# """
#
# """
# 模块作用：
# 判断该命令是否成功
# """
#
# import logging
# import os
# import sys
# import subprocess
# import datetime as dt
#
# """
# log init
# """
#
# program = os.path.basename(sys.argv[0])
# logger = logging.getLogger(program)
# logging.basicConfig(format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s',
#                     datefmt='%a, %d %b %Y %H:%M:%S')
# logging.root.setLevel(level=logging.INFO)
#
# def check_process_stat(ex_code, exc_str="exc"):
#     if ex_code<>0:
#         logger.error("""{exc_str} is fail""".format(exc_str = exc_str))
#         exit(1)
#     else:
#         logger.info("""{exc_str} is sucess""".format(exc_str = exc_str))
#
# def do_exec(cmd,limit_count=4):
#     """
#     判断该命令是否成功
#     输入参数：
#     cmd：linux命令
#     limit_count:命令重提数(默认为4)
#     返回参数：
#     code:命令是否成功 若成功=0 不成功 <>0
#
#     note:
#     函数中会返回标准输出：用于日志输出
#     """
#     code =1
#     count=0
#
#     while ((code <> 0) and (count < limit_count)):
#         #执行传入的linux命令:cmd
#         ## print "{t} exec the cmd:\n {c}".format( c = cmd, t=dt.datetime.now().strftime("%Y%m%d-%H:%M:%S") )
#         logger.info("exec the cmd:\n {c}".format( c = cmd ))
#         subp=subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE)
#         #等待linux命令执行完成
#         subp.wait()
#         #得到程序返回码：
#         code=subp.returncode
#         #输出执行命令后的内容
#         for line in subp.stdout.readlines():
#             # print line
#         count += 1
#         ## print 'states_code:',code
#         ## print 'circle_count:',count
#         logger.info('states_code:'+str(code))
#         logger.info('circle_count:'+str(count))
#         ## print 'circle_count:',count
#         ## print "\n"
#     return code
#
#
#
#
#
#
