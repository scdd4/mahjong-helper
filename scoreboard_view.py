"""Orient scoreboard crops so seat assignment is explicit to the vision model."""
from pathlib import Path
import os
from PIL import Image,ImageDraw,ImageFont


def scoreboard_panel(image):
    w,h=image.size
    # Ratios refer to the game client, never desktop coordinates.
    fields=[('自己点棒',(.456,.415,.544,.482),0),
            ('下家点棒',(.528,.337,.566,.458),-90),
            ('对家点棒',(.456,.294,.544,.366),180),
            ('上家点棒',(.437,.337,.471,.458),90),
            ('本家风位',(.408,.450,.454,.503),0)]
    panel=Image.new('RGB',(600,300),'#202736')
    draw=ImageDraw.Draw(panel)
    try:font=ImageFont.truetype(str(Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts/msyh.ttc'),19)
    except OSError:
        font=ImageFont.load_default()
        english=['SELF','RIGHT','ACROSS','LEFT','SELF WIND']
        fields=[(english[i],ratios,rotation) for i,(_,ratios,rotation) in enumerate(fields)]
    for i,(label,ratios,rotation) in enumerate(fields):
        x,y=(i%2)*300,(i//2)*100
        draw.text((x+8,y+3),label,fill='white',font=font)
        rect=tuple(round(v*(w if j%2==0 else h)) for j,v in enumerate(ratios))
        with image.crop(rect) as crop:
            view=crop.rotate(rotation,expand=True)
            try:
                factor=min(280/view.width,65/view.height)
                enlarged=view.resize((max(1,round(view.width*factor)),max(1,round(view.height*factor))),Image.Resampling.LANCZOS)
                panel.paste(enlarged,(x+8,y+29));enlarged.close()
            finally:view.close()
    return panel
