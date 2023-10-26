#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   redis_uits.py
@Time    :   2023/06/28 04:35:27
@Author  :   Li Ruilong
@Version :   1.0
@Contact :   liruilonger@gmail.com
@Desc    :   Redis 工具类
"""

# here put the import lib

import redis
import yaml_utils as Yaml


class RedisClient:
    def __init__(self,db=0):

        db_config = Yaml.get_yaml_config(file_name="config/config.yaml")['queue']['redis']
        #pool = redis.ConnectionPool(host= db_config['host'], port=db_config['port'], password=db_config['password'],db= db,decode_responses=True)
        self.r = redis.Redis(host= db_config['host'], port=db_config['port'], password=db_config['password'],db=  db_config['database'],)
        

# 字符串
    def str_set(self, key, value):
        self.r.set(key, value)
        return self.r

    def str_get(self, key):
        return self.r.get(key)
    

    def delete(self, key):
        self.r.delete(key)
        return self.r
        
# 列表
    def lpush(self, key, value):
        self.r.lpush(key, value)
        return self.r 
        

    def rpush(self, key, value):
        self.r.rpush(key, value)
        return self.r 
    
    def lpop(self, key ):
        
        return self.r.lpop(key)
        

    def rpop(self, key):
       
        return  self.r.rpop(key)
    
    def brpop(self,key,timeout):

        return  self.r.brpop(key,timeout)
    
    def blpop(self,key,timeout):
        return  self.r.blpop(key,timeout)
    
    def lrange(self, key, start, end):
        return self.r.lrange(key, start, end)

# 集合
    def sadd(self, key, value):
        self.r.sadd(key, value)
        return self.r
    def srem(self, key, value):
        self.r.srem(key, value)
        return self.r
    def sismember(self, key, value):
        return self.r.sismember(key, value)

    def smembers(self, key):
        return self.r.smembers(key)
    
    def scard(self,set_name):
        return self.r.scard(set_name)

# 哈希表
    def hset(self, key, field, value):
        self.r.hset(key, field, value)
        return self.r

    def hget(self, key, field):
        return self.r.hget(key, field)
    
    def hdel(self, key, field,boo=False):
        self.r.hdel(key, field)
        if boo:
            return self.r.hgetall(key)
        else:
            return self.r
        
    def hmset(self,key,data):
        self.r.hmset(key,data)

    def hgetall(self,key):
        return self.r.hgetall(key)

    def hlen(self,hash_name):
        size = self.r.hlen(hash_name)
        return size
    
    def  hexists(self,key, field):
        return self.r.hexists(key, field)


# 有序集合
    def zadd(self, key,  member,score):
        self.r.zadd(key, {member: score})
        return self.r 

    def zrange(self, key, start, end):
        return self.r.zrange(key, start, end)
    
    # 获取集合长度
    def zcard(self, key):
        return self.r.zcard(key)
    
    def zadd(self, key,  member):
        self.r.zrem(key, member)
        return self.r 


    
    


    def zrange_w(self, key, start, end):
        return self.r.zrange(key, start, end,withscores=True)

    def zrangebyscore(self, key, min_score, max_score):
        return self.r.zrangebyscore(key, min_score, max_score)

 

    def clear(self):
        self.r.flushall()
        return self.r 
    
    def close(self):
        self.r.close()
        return None
    
    def clear_db(self):
        self.r.flushdb()
        return None
    

# 计数器
    def incr(self,namespect):
        self.r.incr(namespect)

# 管道
    def pipeline(self):
        return self.r.pipeline()

    def get_hash_data_by_pattern(self, pattern):
        cursor = '0'
        hash_data = {}
        
        while cursor != 0:
            cursor, keys = self.r.scan(cursor=cursor, match=pattern)
            
            for key in keys:
                hash_data[key] = self.r.hgetall(key)
        
        return hash_data    
    

def con_test():
    # 创建 Redis 连接对象
    r = redis.Redis(host='192.168.26.55', port=6379, password="123456", db=0)

    # 测试连接是否成功
    try:
        r.ping()
        print('连接成功')
    except redis.exceptions.ConnectionError:
        print('连接失败')



if __name__ == "__main__":
    con_test()
