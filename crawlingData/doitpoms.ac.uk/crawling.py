## crawling data frow "https://www.doitpoms.ac.uk/miclib/browse.php"
import urllib
import urllib2
import re

def mkdir(path): #创建文件目录
    import os
    # 去除首位空格
    path=path.strip()
    # 去除尾部 \ 符号
    path=path.rstrip("\\")
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists=os.path.exists(path)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path) 
        #print path+'success'
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        #print path+'already created'
        return False
    

def eachPage(thirdUrl,fileDir):
    request = urllib2.Request(thirdUrl)
    response = urllib2.urlopen(request)
    content = response.read().decode('utf-8')
    #print content
    
    string3='img src="micrographs/small/(.*?)\.(.*?)" border="0" width="100"'
    thirdPattern = re.compile(string3,re.S)
    items = re.findall(thirdPattern,content)
    for item in items:
        #print item[0]
        imgId = int(item[0])
        fouthUrl = 'https://www.doitpoms.ac.uk/miclib/micrograph.php?id='+ str(imgId)
        #downUrl = 'https://www.doitpoms.ac.uk/miclib/micrographs/large/'+item[0]+'.jpg'
        request2 = urllib2.Request(fouthUrl)
        response2 = urllib2.urlopen(request2)
        content2 = response2.read().decode('utf-8')
        string4='<img src="micrographs/large/(.*?)" border="(0)"'
        fouthPattern = re.compile(string4,re.S)
        itemTemp = re.search(fouthPattern,content2)
        print itemTemp.group(1)
        downUrl = 'https://www.doitpoms.ac.uk/miclib/micrographs/large/'+itemTemp.group(1)
        downPath = fileDir + '/' + itemTemp.group(1)
        urllib.urlretrieve(downUrl, downPath) ##下载图片
    
    
def microImgsListPage(dir1,dir2,urlTemp,urlBase):
    fileDir = 'images/'+dir1+'['+dir2+']'
    mkdir(fileDir)
    secondUrl = urlBase+urlTemp+'&list=mic'
    #print secondUrl
    request = urllib2.Request(secondUrl)
    response = urllib2.urlopen(request)
    #print response.read()
    content = response.read().decode('utf-8')
    if int(dir2) > 10:
        string2='Page <strong>1</strong> <a.*?of results">(\d*)</a> \| Previous \|'
        secondPattern = re.compile(string2,re.S)
        item = re.search(secondPattern,content)
        pageNum = int(item.group(1))
        pageNum +=1
        #print pageNum
        for i in range(1,pageNum):
            pageUrl = secondUrl + '&page='+ str(i)
            #print pageUrl
            eachPage(pageUrl,fileDir)
    else:
        eachPage(secondUrl,fileDir)
        

##########################################################################################
baseUrl = 'https://www.doitpoms.ac.uk/miclib/browse.php'
request = urllib2.Request(baseUrl)
response = urllib2.urlopen(request)
#print response.read()
content = response.read().decode('utf-8')

#re.S:点任意匹配模式; .*? 匹配任意无限多个字符
string1='<small><strong>(.*?)</strong>\s\[(\w+)\]</small></dt>.<dd><small>Show relevant:.*?>keywords</a>,  <a href="browse.php(.*?)&amp;list=mic" title="List micrographs for this category">micrographs</a></small><br /><br /></dd>'
basePattern = re.compile(string1,re.S)
items = re.findall(basePattern,content)
for item in items:
   #print item[0],item[1],item[2]
    microImgsListPage(item[0],item[1],item[2],baseUrl)

print 'done'
    

