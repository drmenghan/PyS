__author__ = 'mhan7'
import urllib.request
FileNameNum=1
herf='index.html'
while True:
    f=open(herf,'w')
    urlh='http://sebug.net/paper/python/'
    url=urlh+herf
    print(url)
    user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
    headers={'User-Agent':user_agent,}
    request=urllib.request.Request(url,None,headers)
    response = urllib.request.urlopen(request)

    # response=urllib.request.urlopen("http://grid.cs.gsu.edu/~mhan7")
    html=response.read()
    print(html)
    NameEnd=html.find('обр╩рЁ')-2
    if NameEnd < 0:
        break
    NameStrat=html.find('href',NameEnd-20)+6
    herf=html[NameStrat:NameEnd]
    print(herf)
    f.write(html)
    f.close()
    FileNameNum+=1
