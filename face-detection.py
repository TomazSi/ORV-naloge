import cv2 as cv
import numpy as np

def zmanjsaj_sliko(slika, sirina, visina):
    '''Zmanjšaj sliko na velikost sirina x visina.'''
    return cv.resize(slika,(sirina,visina))

def obdelaj_sliko_s_skatlami(slika, sirina_skatle, visina_skatle, barva_koze) -> list:
    '''Sprehodi se skozi sliko v velikosti škatle (sirina_skatle x visina_skatle) in izračunaj število pikslov kože v vsaki škatli.
    Škatle se ne smejo prekrivati!
    Vrne seznam škatel, s številom pikslov kože.
    Primer: Če je v sliki 25 škatel, kjer je v vsaki vrstici 5 škatel, naj bo seznam oblike
      [[1,0,0,1,1],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[1,0,0,0,1]]. 
      V tem primeru je v prvi škatli 1 piksel kože, v drugi 0, v tretji 0, v četrti 1 in v peti 1.'''
    visina,sirina,_=slika.shape #shrani visino in sirino
    skatle=[]
    for y in range(0,visina,visina_skatle):
        vrsta=[]
        for x in range(0,sirina,sirina_skatle):
            skatla=slika[y:y+visina_skatle,x:x+sirina_skatle] #izreze skatlo
            piksli_koze=prestej_piklse_z_barvo_koze(skatla,barva_koze)
            vrsta.append(piksli_koze)
        skatle.append(vrsta)
    return skatle

def prestej_piklse_z_barvo_koze(slika, barva_koze) -> int:
    '''Prestej število pikslov z barvo kože v škatli.'''
    lower=np.array(barva_koze[0],dtype="uint8")
    upper=np.array(barva_koze[1],dtype="uint8")
    maska=cv.inRange(slika,lower,upper)
    piksli_koze=cv.countNonZero(maska)
    return piksli_koze

def pobarvaj_skozi_povezane_kocke(slika, povezane_kocke):
    '''Pobarva kvadrate na sliki glede na povezanost.'''
    barve = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]  # definiramo različne barve
    for k, skupina in enumerate(povezane_kocke):
        barva = barve[k % len(barve)]  # izberemo barvo za skupino
        for kocka in skupina:
            cv.rectangle(slika, (kocka[1]*50, kocka[0]*50), (kocka[1]*50+50, kocka[0]*50+50), barva, 2)

def doloci_barvo_koze(slika,levo_zgoraj,desno_spodaj) -> tuple:
    '''Ta funkcija se kliče zgolj 1x na prvi sliki iz kamere. 
    Vrne barvo kože v območju ki ga definira oklepajoča škatla (levo_zgoraj, desno_spodaj).
      Način izračuna je prepuščen vaši domišljiji.'''
    bbox = cv.selectROI("Izberi območje za barvo kože", slika, False)
    # Izrezemo obmocje
    obmocje = slika[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])]
    # spodnja/zgornja meja
    spodna_meja = np.array([np.min(obmocje[:,:,0])-10, np.min(obmocje[:,:,1])-10, np.min(obmocje[:,:,2])-10])
    zgorna_meja = np.array([np.max(obmocje[:,:,0])+10, np.max(obmocje[:,:,1])+10, np.max(obmocje[:,:,2])+10])
    cv.destroyAllWindows()
    return spodna_meja, zgorna_meja

if __name__ == '__main__':
    #Pripravi kamero
    camera=cv.VideoCapture(0)
    if not camera.isOpened():
        print("Kamera ni na voljo")
        exit()
    #Zajami prvo sliko iz kamere
    ret, prva_slika=camera.read()
    if not ret:
        print("Napaka pri zajemu slike")
        exit()
    prva_slika=zmanjsaj_sliko(prva_slika,320,240)
    flipprva=cv.flip(prva_slika,1)
    #Izračunamo barvo kože na prvi sliki
    levo_zgoraj=(140,120)
    desno_spodaj=(180,150)
    spodna_meja,zgorna_meja=doloci_barvo_koze(prva_slika,levo_zgoraj,desno_spodaj)
    #Zajemaj slike iz kamere in jih obdeluj     
    while True:
        #Za merjenje hitrosti kamere
        start_time = cv.getTickCount()
        #zajem slike
        ret, slika=camera.read()
        if not ret:
            print("Napaka pri zajemu slike")
            break
        #zmanjšaj
        slika=zmanjsaj_sliko(slika,320,240)
        #skatle
        skatle=obdelaj_sliko_s_skatlami(slika, 50,50,(spodna_meja,zgorna_meja))

        # seznam kock
        povezane_kocke = []
        for y, vrsta in enumerate(skatle):
            for x, piksli_koze in enumerate(vrsta):
                if piksli_koze > 100:  # nastavlanje
                    je_povezan = False
                    for skupina in povezane_kocke:
                        for kocka in skupina:
                            # ali je kvadrat v s skupino
                            if abs(kocka[0] - y) <= 1 and abs(kocka[1] - x) <= 1:
                                skupina.append((y, x))
                                je_povezan = True
                                break
                        if je_povezan:
                            break
                    if not je_povezan:
                        # nova skupina
                        povezane_kocke.append([(y, x)])
        pobarvaj_skozi_povezane_kocke(slika, povezane_kocke)

        # Za izračun hitrosti kamere https://stackoverflow.com/questions/72569986/opencv-limit-fps-while-running
        fps = cv.getTickFrequency() / (cv.getTickCount() - start_time)

        flipslika=cv.flip(slika,1) #https://www.tutorialspoint.com/how-to-flip-an-image-in-opencv-python
        # Dodatek: Izpis hitrosti kamere
        cv.putText(flipslika, f"FPS: {int(fps)}", (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv.imshow("Camera", flipslika)
        if cv.waitKey(1)&0xFF==ord('q'):
            break

    #Označi območja (škatle), kjer se nahaja obraz (kako je prepuščeno vaši domišljiji)
        #Vprašanje 1: Kako iz števila pikslov iz vsake škatle določiti celotno območje obraza (Floodfill)?
        #Vprašanje 2: Kako prešteti število ljudi?

        #Kako velikost prebirne škatle vpliva na hitrost algoritma in točnost detekcije? Poigrajte se s parametroma velikost_skatle
        #in ne pozabite, da ni nujno da je škatla kvadratna.
    pass