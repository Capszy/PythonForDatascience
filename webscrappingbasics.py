from bs4 import BeautifulSoup

html_doc='...'
soup=BeautifulSoup(html_doc,'html.parser')
print(soup)

print soup.prettify()(0:350)

soup=BeautifulSoup('<b body="description">Product Description</b>','html')
tag=soup.b
type(tag)

print tag
tag.name
tag.name='bestbooks'
tag
tag.name

tag('body')
tag.attrs
tag['id']=3
tag.attrs
del tag['body']

soup.head
soup.title
soup.body.b
soup.body
soup.ul
soup.a

tag=soup.b
type(tag)
tag.name
tag.string
type(tag.string)
nav_string=tag.string
nav_string.replace_with('Null')
tag.string

for string in soup.stripped_strings: print(repr(string))
title_tag=soup.title
title_tag
title-tag.parent
title_tag.string
title_tag.string.parent

import re#regular expression

soup=BeautifulSoup(r,'lxml')
type(soup)

print soup.prettify()(0:100)
text_onlysoup.get_text()
print(text_only)
soup.find_all=("li")
soup.find_all(['ul','b'])
l=re.compile('l')
for tag in soup.find_all(l): print(tag.name)
for tag in soup.find_all(True): print(tag.name)
for link in soup.find_all('a'): print (link.get('href'))
soup.find_all(string=re.compile("data"))


r=urllib.urlopen('https://analytics.usa.gov').read()
soup=BeautifulSoup(r,"lxml")
type(soup)
print soup.prettify()[:100]
for link in soup.find_all('a'):print link.get('href')
for link in soup.findAll('a',attrs={'href':re.compile("^http")}):print link
file=open('parsed_data.txt','wb')
for link in soup.findAll('a',attrs={'href':re.compile("^http")}):
    soup_link=str(link)
    print soup_link
    file.flush()
    file.close()
    
%pwd # to find where the file is

