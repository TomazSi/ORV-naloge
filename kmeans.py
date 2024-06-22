import cv2 as cv
import numpy as np

def kmeans(slika, k, iteracije=10):
    '''Izvede segmentacijo slike z uporabo metode k-means.'''
    # Izračunamo začetne centre
    centri = izracunaj_centre(slika, izbira='nakljucna', dimenzija_centra=k, T=10)
    
    # Pridobimo dimenzije slike
    vrstice, stolpci, _ = slika.shape
    # Pretvorimo sliko v 2D matriko
    slika_2D = np.reshape(slika, (vrstice * stolpci, 3))
    
    for _ in range(iteracije):
        # Izračunamo razdalje od vsake točke do centrov
        razdalje = np.sum(np.abs(slika_2D[:, None, :] - centri[None, :, :]), axis=2)
        # Najdemo indekse centrov, ki so najbližje vsaki točki
        indeksi_centrov = np.argmin(razdalje, axis=1)
        # Posodobimo centre na osnovi povprečja vseh točk, ki so jim dodeljeni
        for i in range(k):
            tocke = slika_2D[indeksi_centrov == i]
            if len(tocke) > 0:
                centri[i] = np.mean(tocke, axis=0)
    
    # Ustvarimo segmentirano sliko
    segmentirana_slika = np.reshape(centri[indeksi_centrov], (vrstice, stolpci, 3))
    
    return segmentirana_slika




def izracunaj_centre(slika, izbira, dimenzija_centra, T):
    '''Izračuna centre za metodo kmeans.'''
    # Pridobimo dimenzije slike
    vrstice, stolpci, _ = slika.shape
    # Pretvorimo sliko v 2D matriko
    slika_2D = np.reshape(slika, (vrstice * stolpci, 3))

    if izbira == 'rocno':
        # Ročna izbira centrov
        slika_copy = slika.copy()
        centri = []
        
        def klik(event, x, y, flags, param):
            if event == cv.EVENT_LBUTTONDOWN:
                centri.append(slika[y, x].tolist())
                cv.circle(slika_copy, (x, y), 5, (0, 255, 0), -1)
                cv.imshow('Izbira centrov', slika_copy)
        
        cv.imshow('Izbira centrov', slika_copy)
        cv.setMouseCallback('Izbira centrov', klik)
        
        while True:
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        cv.destroyAllWindows()
        centri = np.array(centri)
        
    elif izbira == 'nakljucna':
        # Naključna izbira centrov
        if dimenzija_centra == 3:  # Only color
            centri = np.random.randint(0, 256, size=(dimenzija_centra, 3))
        elif dimenzija_centra == 5:  # Location and color
            centri = np.random.randint(0, 256, size=(dimenzija_centra, 3))  # Use only color for centers
        
        # Preverimo, ali so centri dovolj oddaljeni
        while True:
            razdalje = np.sum(np.abs(centri[:, None, :] - centri[None, :, :]), axis=2)
            razdalje = razdalje.astype(float)  # Convert to floating-point array
            np.fill_diagonal(razdalje, np.inf)
            if np.all(razdalje > T):
                break
            else:
                # Ponovno naključno izberemo centre
                if dimenzija_centra == 3:
                    centri = np.random.randint(0, 256, size=(dimenzija_centra, 3))
                elif dimenzija_centra == 5:
                    centri = np.random.randint(0, vrstice, size=(dimenzija_centra, 2))
                    centri = np.concatenate((centri, np.random.randint(0, 256, size=(dimenzija_centra, 3))), axis=1)
    
    return centri

if __name__ == "__main__":
    lenna_slika = cv.imread(".utils/lenna.png")
    zelenjava_slika = cv.imread(".utils/zelenjava.jpg")
    # Testiranje algoritma k-means
    lenna_segmentirana = kmeans(lenna_slika, k=3, iteracije=10)
    
    # Prikaži rezultate
    cv.imshow("Lenna segmentirana", lenna_segmentirana.astype(np.uint8)) # Pretvori v tip uint8
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    zelenjava_segmentirana = kmeans(zelenjava_slika, k=3, iteracije=10)
    cv.imshow("Zelenjava segmentirana", zelenjava_segmentirana.astype(np.uint8)) # Pretvori v tip uint8
    cv.waitKey(0)
    cv.destroyAllWindows()
