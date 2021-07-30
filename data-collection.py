import pandas as pd
import requests
from bs4 import BeautifulSoup
import shutil, sys  

page = requests.get("https://pokemondb.net/pokedex/national")
soup = BeautifulSoup(page.content, 'html.parser')

for card in soup.find_all('div', class_='infocard'):
    image = card.find_all(class_='img-sprite')[0]
    name = card.find_all('a', class_='ent-name')[0].get_text()
    itype = card.find_all('a', class_='itype')
    for t in itype:
        name=name+"_"+t.get_text()
    r = requests.get(image.attrs['data-src'], stream=True)
    if r.status_code == 200:                 
        with open(f"images/{name}.jpg", 'wb') as f: 
            r.raw.decode_content = True
            shutil.copyfileobj(r.raw, f)
