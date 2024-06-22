import cv2 as cv
import numpy as np
import math

def konvolucija(slika, jedro):
    '''Izvede konvolucijo nad sliko. Brez uporabe funkcije cv.filter2D, ali katerekoli druge funkcije, ki izvaja konvolucijo.
    Funkcijo implementirajte sami z uporabo zank oz. vektorskega računanja.'''
    visina_slike, sirina_slike = len(slika), len(slika[0])
    velikost_jedra = len(jedro)
    pol_velikosti_jedra = velikost_jedra // 2
    nova_visina = visina_slike - 2 * pol_velikosti_jedra
    nova_sirina = sirina_slike - 2 * pol_velikosti_jedra
    nova_slika = [[0 for _ in range(nova_sirina)] for _ in range(nova_visina)]

    # dodajanje paddinga
    slika_pad = dodaj_padding(slika, pol_velikosti_jedra)

    # izvedba konvolucije
    for y in range(pol_velikosti_jedra, visina_slike - pol_velikosti_jedra):
        for x in range(pol_velikosti_jedra, sirina_slike - pol_velikosti_jedra):
            vsota = 0
            for j in range(velikost_jedra):
                for i in range(velikost_jedra):
                    vsota += slika_pad[y + j - pol_velikosti_jedra][x + i - pol_velikosti_jedra] * jedro[j][i]
            nova_slika[y - pol_velikosti_jedra][x - pol_velikosti_jedra] = vsota
    return nova_slika

def dodaj_padding(slika, pol_velikosti_jedra):
    visina, sirina = slika.shape
    nova_visina = visina + 2 * pol_velikosti_jedra
    nova_sirina = sirina + 2 * pol_velikosti_jedra
    slika_z_paddingom = np.zeros((nova_visina, nova_sirina), dtype=slika.dtype)

    # kopiranje izvirne slike v sredino nove slike
    slika_z_paddingom[pol_velikosti_jedra:-pol_velikosti_jedra, pol_velikosti_jedra:-pol_velikosti_jedra] = slika

    return slika_z_paddingom

def filtriraj_z_gaussovim_jedrom(slika,sigma):
    '''Filtrira sliko z Gaussovim jedrom..'''
    velikost_jedra = int((2 * sigma) * 2 + 1)
    pol_velikosti_jedra = velikost_jedra // 2
    jedro = np.zeros((velikost_jedra, velikost_jedra))

    for y in range(velikost_jedra):
        for x in range(velikost_jedra):
            i = y - pol_velikosti_jedra
            j = x - pol_velikosti_jedra
            jedro[y, x] = 1 / (2 * math.pi * sigma**2) * math.exp(-(i**2 + j**2) / (2 * sigma**2))

    return konvolucija(slika, jedro)

def filtriraj_sobel_smer(slika):
    '''Filtrira sliko z Sobelovim jedrom in označi gradiente v orignalni sliki glede na ustrezen pogoj.'''
    sobel_x = [[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]]

    gradient_x = konvolucija(slika, sobel_x)

    nova_visina = len(gradient_x)
    nova_sirina = len(gradient_x[0])
    magnitude = [[0 for _ in range(nova_sirina)] for _ in range(nova_visina)]

    for y in range(nova_visina):
        for x in range(nova_sirina):
            magnitude[y][x] = abs(gradient_x[y][x])

    threshold = 230
    slika_marked = []

    for y in range(nova_visina):
        vrstica = []
        for x in range(nova_sirina):
            if magnitude[y][x] > threshold:
                vrstica.append([0, 0, 255])  # Set strong gradients to red color
            else:
                vrstica.append([0, 0, 0])  # Set other pixels to black
        slika_marked.append(vrstica)

    return np.array(slika_marked, dtype=np.uint8)
if __name__ == '__main__':    
    slike = []

    # Prva slika
    slika1 = np.zeros((4, 4), dtype=np.uint8)
    slika1[0, 0] = 1
    slika1[1, 1] = 2
    slika1[2, 2] = 3    
    slika1[3, 3] = 4

    # Druga slika
    slika2 = np.zeros((4, 4), dtype=np.uint8)
    slika2[0, 0] = 1
    slika2[1, 1] = 1
    slika2[2, 2] = 1    
    slika2[3, 3] = 1

    # Tretja slika
    slika3 = np.zeros((4, 4), dtype=np.uint8)
    slika3[1, 1] = 1

    slike = [slika1, slika2, slika3]

    # Izvedemo konvolucijo slike z jedrom
    jedro = np.array([[1, 1, 1],
                     [1, 1, 1],
                     [1, 1, 1]])

    # Izvedemo konvolucijo
    for slikaa in slike:
        slika_konvolucija = cv.filter2D(slikaa, -1, jedro,borderType=cv.BORDER_CONSTANT)
        # Prikažemo izvirno sliko
        print("Original:\n {}".format(slikaa))
        print("Konvolucija:\n {}".format(slika_konvolucija))
        print("######################")
    
    slika = cv.imread('.utils/lenna.png')
    slika_siva = cv.cvtColor(slika, cv.COLOR_BGR2GRAY)

    filtrirana_slika = filtriraj_z_gaussovim_jedrom(slika_siva, 2)
    cv.imshow('Filtrirana slika s sigmo', np.array(filtrirana_slika, dtype=np.uint8))

    slika_filtrirana = filtriraj_sobel_smer(slika_siva.copy()) 
    cv.imshow('Slika z oznacenimi mocnimi gradienti', np.array(slika_filtrirana, dtype=np.uint8))
    cv.waitKey(0)
    