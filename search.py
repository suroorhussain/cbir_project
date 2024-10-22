import colordescriptor
import searcher
import argparse
import cv2

def main(query, index, result, main=False):
    #load the query image
    query = cv2.imread(query)

    #perform the search
    s = searcher.Searcher(index)
    results = s.search(query)

    """for (result, idd) in results:
        cv2.imshow("Result", result)
        cv2.waitKey(0)"""

    #display the query
    cv2.namedWindow('Query', cv2.WINDOW_NORMAL)
    cv2.imshow("Query", query)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #loop over the results

    cv2.namedWindow('Result', cv2.WINDOW_NORMAL)

    for imgpath in results:
        #print "ResultID ", resultID
        #result = cv2.imread(args["result_path"] + "/" + resultID)
        img = cv2.imread(imgpath)
        cv2.imshow("Result", img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--index", required = True,
                    help = "Path to where the computed index will be stored")
    ap.add_argument("-q", "--query", required = True,
                    help = "Path to the query image")
    ap.add_argument("-r", "--result-path", required = True,
                    help = "Path to the result path")
    args = vars(ap.parse_args())
    main(args['query'], args['index'], args['result_path'], main=True)
