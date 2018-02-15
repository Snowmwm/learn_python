#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import hashlib
import json
from textwrap import dedent
from time import time
from urllib.parse import urlparse
from uuid import uuid4
import requests
from flask import Flask, jsonify, request


class Blockchain(object):
    def __init__(self):
        self.chain = []
        self.current_transactions = []
        self.nodes = set()
        
        # 生成创世区块
        self.new_block(previous_hash=1, proof=100)

    def new_block(self, proof, previous_hash=None):
        """
        创建一个新的区块到区块链中
        proof: <int>由工作证明算法生成的证明
        previous_hash: (Optional)<str>前一个区块的hash值
        return: <dict>新区块
        """
        block = {
            'index': len(self.chain) + 1,
            'timestamp': time(),
            'transactions': self.current_transactions,
            'proof': proof,
            'previous_hash': previous_hash or self.hash(self.chain[-1]),
            }
            
        # 重置当前交易记录
        self.current_transactions = []
        
        self.chain.append(block)
        return block

    def new_transaction(self, sender, recipient, amount):
        """
        将一个新的交易添加到下一个被挖掘的区块的事物列表
        sender: <str> 发送人地址 
        recipient: <str> 接收人地址
        amount: <int> 金额
        return: <int> 包含本次交易的区块的索引
        """
        self.current_transactions.append({
            'sender': sender,
            'recipient': recipient,
            'amount': amount,
        })
        return self.last_block['index'] + 1

    @staticmethod
    #@staticmethod装饰器将类中的函数定义成静态方法
    def hash(block):
        """
        计算一个区块的hash
        block: <dict> 区块
        return: <str> 区块的hash
        """
        block_string = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

    @property
    #@property装饰器将一个方法变成属性调用的
    def last_block(self):
        #返回区块链中的最后一个区块
        return self.chain[-1]
        
    def proof_of_work(self, last_block):
        """
        找到一个数字P,使得它与前一个区块的proof拼接成的字符串的Hash值以4 个零开头。
        last_block: <dict> 前一个区块
        return: <int> 当前区块的Proof
        """

        last_proof = last_block['proof']
        last_hash = self.hash(last_block)

        proof = 0
        while self.valid_proof(last_proof, proof, last_hash) is False:
            proof += 1
        return proof

    @staticmethod
    def valid_proof(last_proof, proof, last_hash):
        """
        验证proof
        last_proof: <int> 前一个区块的proof
        proof: <int> 当前区块的Proof
        last_hash: <str> 前一个区块的hash
        :return: <bool> 是否通过验证，是返回True，否返回Flase
        """
        guess = f'{last_proof}{proof}{last_hash}'.encode()
        guess_hash = hashlib.sha256(guess).hexdigest()
        return guess_hash[:4] == "0000"
        
    def register_node(self, address):
        """
        向节点列表添加一个新的节点
        address: <str> 节点的地址 Eg. 'http://192.168.0.5:5000'
        return: None
        """

        parsed_url = urlparse(address)
        if parsed_url.netloc:
            self.nodes.add(parsed_url.netloc)
        elif parsed_url.path:
            # Accepts an URL without scheme like '192.168.0.5:5000'.
            self.nodes.add(parsed_url.path)
        else:
            raise ValueError('Invalid URL')
            
    def valid_chain(self, chain):
        """
        通过遍历每个区块并验证hash和pow，检查一个区块链是否有效
        chain: <list> 区块链
        return: <bool> 有效返回True，无效返回False
        """

        last_block = chain[0]
        current_index = 1

        while current_index < len(chain):
            block = chain[current_index]
            print(f'{last_block}')
            print(f'{block}')
            print("\n-----------\n")
            #验证hash
            if block['previous_hash'] != self.hash(last_block):
                return False

            #验证工作量证明
            if not self.valid_proof(last_block['proof'], block['proof'],
                                    last_block['previous_hash']):
                return False

            last_block = block
            current_index += 1

        return True

    def resolve_conflicts(self):
        """
        共识算法。
        return: <bool> 如果我们的链被替代了，返回True，否则返回Flase
        """

        neighbours = self.nodes
        new_chain = None

        max_length = len(self.chain)

        # 遍历我们所有邻居节点，抓取区块链并验证
        for node in neighbours:
            response = requests.get(f'http://{node}/chain')

            if response.status_code == 200:
                length = response.json()['length']
                chain = response.json()['chain']

                if length > max_length and self.valid_chain(chain):
                    max_length = length
                    new_chain = chain

        # 如果找到一个长度大于我们的有效链，就用它取代我们的链。
        if new_chain:
            self.chain = new_chain
            return True

        return False         
        
#实例化节点
app = Flask(__name__)

#为节点创建一个随机的名称
node_identifier = str(uuid4()).replace('-', '')

#实例化 Blockchain 类
blockchain = Blockchain()

#创建 /mine 接口，GET 方式请求， 告诉服务器去挖掘新的区块
@app.route('/mine', methods=['GET'])
def mine():
    #计算工作量证明
    last_block = blockchain.last_block
    #last_proof = last_block['proof']
    proof = blockchain.proof_of_work(last_block)

    #通过新增一笔交易奖励矿工一定数量的币
    #交易发起人是"0"表示这个节点开采出一个新的币
    blockchain.new_transaction(
        sender="0",
        recipient=node_identifier,
        amount=1,
    )

    #创建新的区块并将其添加到区块链中
    previous_hash = blockchain.hash(last_block)
    block = blockchain.new_block(proof, previous_hash)

    response = {
        'message': "New Block Forged",
        'index': block['index'],
        'transactions': block['transactions'],
        'proof': block['proof'],
        'previous_hash': block['previous_hash'],
    }
    return jsonify(response), 200

#创建 /transactions/new 接口，POST 方式请求，可以给接口发送交易数据
@app.route('/transactions/new', methods=['POST'])
def new_transaction():
    values = request.get_json()

    # 检查给接口发送的交易数据中是否有required中的字段。
    required = ['sender', 'recipient', 'amount']
    if not all(k in values for k in required):
        return 'Missing values', 400

    # 添加一个新的交易
    index = blockchain.new_transaction(values['sender'], values['recipient'], values['amount'])

    response = {'message': f'Transaction will be added to Block {index}'}
    return jsonify(response), 201 

#创建 /chain 接口，GET 方式请求，返回整个区块链
@app.route('/chain', methods=['GET'])
def full_chain():
    response = {
        'chain': blockchain.chain,
        'length': len(blockchain.chain),
    }
    return jsonify(response), 200

#创建 /nodes/register 接口，POST方式请求，接收 URL 形式的新节点列表
@app.route('/nodes/register', methods=['POST'])
def register_nodes():
    values = request.get_json()

    nodes = values.get('nodes')
    if nodes is None:
        return "Error: Please supply a valid list of nodes", 400

    for node in nodes:
        blockchain.register_node(node)

    response = {
        'message': 'New nodes have been added',
        'total_nodes': list(blockchain.nodes),
    }
    return jsonify(response), 201

#创建 /nodes/register 接口，GET方式请求，
#执行一致性算法，解决冲突，确保节点拥有正确的链
@app.route('/nodes/resolve', methods=['GET'])
def consensus():
    replaced = blockchain.resolve_conflicts()

    if replaced:
        response = {
            'message': 'Our chain was replaced',
            'new_chain': blockchain.chain
        }
    else:
        response = {
            'message': 'Our chain is authoritative',
            'chain': blockchain.chain
        }

    return jsonify(response), 200
    
#服务器运行端口 5000 
if __name__ == '__main__':
    #app.run(host='0.0.0.0', port=5000)
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-p', '--port', default=5000, type=int, help='port to listen on')
    args = parser.parse_args()
    port = args.port

    app.run(host='0.0.0.0', port=port)