#import qrcode and pyzbar for generating QR it only requires qrcode for generating QR and to decode it we need pyzbar
import qrcode
from pyzbar.pyzbar import decode 
from PIL import Image

#In myqr you can write your own link and can generate qr accordingly
myqr = qrcode.make("https://www.linkedin.com/in/utkarsh-singh-lh999052004/")
myqr.save("myqr.png", scale = 9)

b = decode(Image.open("myqr.png"))
print(b[0].data.decode("ascii"))