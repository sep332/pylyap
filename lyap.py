import operator, cProfile, Image
from math import isnan, log, copysign
from multiprocessing import Pool
from itertools import chain

pattern = ( 0, 0, 1, 0, 1 ) # 'a' = 0, 'b' = 1
xpixels, ypixels = 400, 400 # size of the image in pixels
x1, y1, x2, y2 = 2.0, 4.0, 4.0, 2.0 # numeric range to be rendered
xres = ( x2 - x1 ) / xpixels
xpoints = [ xres * p for p in range( int( x1 / xres ), int( x2 / xres ) ) ]
yres = ( y2 - y1 ) / ypixels
ypoints = [ yres * p for p in range( int( y1 / yres ), int( y2 / yres ) ) ]
coords = [ [ ( a, b ) for a in xpoints ] for b in ypoints ]
numprocs = 6
iterations = 300
pattern *= ( ( iterations / len( pattern ) ) + 1 )
skip = 50 # Let the system "settle" before we start counting
partition_size = 200 # bigger is faster, but too big overflows
x0 = 0.05
gamma = 0.8 # for coloring

def drawLayer():
    grayscale = Image.new( "L", ( xpixels, ypixels ) )
    pool = Pool( numprocs )
    values = []
    values = pool.map( lyap_picklable, coords ) # main calculation
    grayscale.putdata( list( chain.from_iterable(values) ) )
    grayscale.save( "gray52.png", "PNG" )

def lyap( seq, point, n ):
    zscale = 1.0 / ( n - skip - 1 ) # the divisor for averaging
    lpattern = pattern
    partition = skip
    x = x0
    total = 0
    product = 1
    avg = 0
    for i in range( skip ):  # warmup (unused)
        k = point[ lpattern[ i ] ]
        x = k * x * ( 1 - x )
    while ( partition < n ):
        for i in range( partition, min( partition + partition_size, n ) ):
            k = point[ lpattern[ i ] ]
            t = k * x
            x = t * ( 1 - x )
            product *= k - 2 * t
            ##total += log( abs( k - 2 * k * x ) )
        try:
            total += log( abs( product ) )
            product = 1
            partition += partition_size
        except ValueError:
            return 0
    avg = total * zscale
    return 128 * ( copysign( abs( avg ) ** gamma, avg ) ) + 127

def lyap_picklable( line ):
    lpattern, literations = pattern, iterations
    return [ lyap( lpattern, point, literations ) for point in line ]

if __name__ == '__main__':
    cProfile.run( "drawLayer()" )
