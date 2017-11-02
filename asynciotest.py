#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading
import asyncio

async def hello():
    print('hello world! (%s)' % threading.currentThread())
    await asyncio.sleep(1)
    print('hello again! (%s)' % threading.currentThread())
    
loop = asyncio.get_event_loop()
tasks = [hello(), hello()]
loop.run_until_complete(asyncio.wait(tasks))
loop.close()