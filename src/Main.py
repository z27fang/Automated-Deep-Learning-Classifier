from GoogleImageScraper import *


def main(argv):
    query = ''
    number = ''
    dest = 'image'
    opts, args = getopt.getopt(argv,"q:n:d:",["query=","number=", "dest="])

    for opt, arg in opts:
        if opt == '-h':
            print('Usage: python3 Main.py -q <query text> -n <number of images> -d <destination path>')
        elif opt in ("-q", "--query"):
            query = arg
        elif opt in ("-n", "--number"):
            number = arg
        elif opt in ("-d", "--dest"):
            dest = arg
    GoogleImageScraper(query, dest, int(number))

if __name__== '__main__':
    try:
        main(sys.argv[1:])
    except getopt.GetoptError:
        print('python3 Main.py -q <query text> -n <number of images>')
    except:
        e = sys.exc_info()
        print("Error: %s" % e)
    