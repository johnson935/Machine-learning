#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 16:48:43 2018

@author: johuson
"""

a = [[1,2],[3,4],[5,6],[7,8],[9,10]]
# appends the array automatically if you declare the for loop inside square brackets
# list comprehension
 #same as
#x = []
#for i in a:
#   x.append(i[1])
    
    
x = [i[1] for i in a]
    
