# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 12:05:29 2023

@author: admin
"""

import pymysql
import pandas as pd

class HandleDB():

    def __init__(self):
    
        self.conn = pymysql.connect(host='', 
                        port=, db='', user='',
                        passwd='', autocommit=True, charset='utf8')
        
    def get_data(self):
        with self.conn.cursor() as curs:
                    
            # 상장목록 DB 읽기
            sql = "SELECT * FROM chat.QA;"
            curs.execute(sql)
            codes = curs.fetchall()
            
            return codes
        

    
    def find_answer(self, Q):
    
        with self.conn.cursor() as curs:
                    
            # 상장목록 DB 읽기
            sql = "SELECT answer FROM chat.QA where question='{}';".format(Q)
            curs.execute(sql)
            codes = curs.fetchall()
            
            return codes[0][0]
        
    # 소멸자 : DB 연결 해제        
        def __del(self):
            
            self.conn.close()