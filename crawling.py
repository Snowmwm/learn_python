#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from getpage import get_page
from pyquery import PyQuery as pq

#创建抽取代理的元类
#在FreeProxyGetter类中加入__CrawlFunc__和__CrawlFuncCount_，分别表示爬虫函数和爬虫函数的数量
class ProxyMwraclass(type):
    def __new__(cls, name, bases, attrs):
        count = 0
        attrs['__CrawlFunc__'] = []
        attrs['__CrawlName__'] = []
        for k, v in attrs.items():
            if 'crawl_' in k:
                attrs['__CrawlName__'].append(k)
                attrs['__CrawlFunc__'].append(v)
                count += 1
        for k in attrs['__CrawlName__']:
            attrs.pop(k)
        attrs['__CrawlFuncCount__'] = count
        return type.__new__(cls, name, bases, attrs)
        
#创建代理获取类
class ProxyGetter(object, metaclass = ProxyMwraclass):
    def get_raw_proxies(self, site):
        proxies = []
        print('Site', site)
        for func in self.__CrawlFunc__:
            if func.__name__ == site:
                this_page_proxies = func(self)
                for proxy in this_page_proxies:
                    print('Getting', proxy, 'from', site)
                    proxies.append(proxy)
        return proxies
        
    def crawl_daili66(self, page_count = 4):
        start_url = 'http://www.66ip.cn/{}html'
        urls = [start_url.format(page) for page in range(1, page_count + 1)]
        for url in urls:
            print('Crawling', url)
            html = get_page(url)
            if html:
                doc = pq(html)
                trs = doc('.containerbox table tr:gt(0)').items()
                for tr in trs:
                    ip = line.find('.td:nth-child(1)').text()
                    port = line.find('.td:nth-child(2)').text()
                    yield ':'.join([ip, port])
                    
    def crawl_proxy360(self):
        start_url = 'http://www.proxy360.cn/Region/China'
        print('Crawing', start_url)
        html = get_page(start_url)
        if html:
            doc = pq(html)
            lines = doc('div[name = "list_proxy_ip"]').items()
            for line in lines:
                ip = line.find('.tbBottomLine:nth-child(1)').text()
                port = line.find('.tbBottomLine:nth-child(2)').text()
                yield ':'.join([ip, port])
                
    def crawl_goubanjia(self):
        start_url = 'http://www.goubanjia.com/free/gngn/index/shtml'
        html = get_page(start_url)
        if html:
            doc = pq(html)
            tds = doc('td.op').items()
            for td in tds:
                td.find('p').remove()
                yield td.text().replace(' ','')
      
if __name__ == '__main__':
    crawler = ProxyGetter()    #实例化ProxyGetter
    print(crawler.__CrawlName__)
    #遍历每一个__CrawlFunc__
    #1、 在ProxyGetter.__CrawlName__上面，获取可以抓取的网址名
    #2、 触发类方法ProxyGetter.get_raw_proxies(site)
    #3、 遍历ProxyGetter.__CrawlFunc__,如果方法名和网址名相同，则执行这一个方法
    #4、 把每个网址获取到的代理整合成数组输出
    for site_label in range(crawler.__CrawlFuncCount__):
        site = crawler.__CrawlName__[site_label]
        myProxies = crawler.get_raw_proxies(site)    #调用实例方法     