#!/usr/bin/python
#
#  Copyright 2002-2019 Barcelona Supercomputing Center (www.bsc.es)
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

# -*- coding: utf-8 -*-

'''Wordcount Block divide'''
import sys
import os
import pickle
import time
from pycompss.api.task import task
from pycompss.api.parameter import *
from pycompss.api.api import compss_wait_on


def read_word(file_object):
    for line in file_object:
        for word in line.split():
            yield word


def read_word_by_word(fp, sizeBlock):
    """Lazy function (generator) to read a file piece by piece in
    chunks of size approx sizeBlock"""
    data = open(fp)
    block = []
    for word in read_word(data):
        block.append(word)
        if sys.getsizeof(block) > sizeBlock:
            yield block
            block = []
    if block:
        yield block


@task(returns=dict)
def wordCount(data):
    partialResult = {}
    for entry in data:
        if entry not in partialResult:
            partialResult[entry] = 1
        else:
            partialResult[entry] += 1
    return partialResult


@task(dic1=INOUT)
def merge_two_dicts(dic1, dic2):
    for k in dic2:
        if k in dic1:
            dic1[k] += dic2[k]
        else:
            dic1[k] = dic2[k]

if __name__ == "__main__":
    pathFile = sys.argv[1]
    resultFile = sys.argv[2]
    sizeBlock = int(sys.argv[3])

    start = time.time()
    result = {}
    for block in read_word_by_word(pathFile, sizeBlock):
        presult = wordCount(block)
        merge_two_dicts(result, presult)
    result = compss_wait_on(result)

    elapsed = time.time() - start
    print("Elapsed Time: " + str(elapsed))

    ff = open(resultFile, 'w')
    ff.write(str(result), ff)
    ff.close()
