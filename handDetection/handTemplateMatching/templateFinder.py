import cv2
import numpy as np

hands = cv2.imread("hands.jpg")
hands_height, hands_width = hands.shape[:2]

cap = cv2.VideoCapture(0)

while True:
	puzzle = cap.read()[1]
	result = cv2.matchTemplate(puzzle, hands, cv2.TM_CCOEFF)
	(_, _, minLoc, maxLoc) = cv2.minMaxLoc(result)
	topLeft = maxLoc
	botRight = (topLeft[0]+hands_width, topLeft[1]+hands_height)
	roi = puzzle[topLeft[1]:botRight[1], topLeft[0]:botRight[0]]

	mask = np.zeros(puzzle.shape, dtype = "uint8")
	puzzle = cv2.addWeighted(puzzle, 0.2, mask, 0.8, 0)
	puzzle[topLeft[1]:botRight[1], topLeft[0]:botRight[0]] = roi

	cv2.imshow("Feed", puzzle)
	k = cv2.waitKey(1) & 0xFF
	if k==27:
		break

cap.release()
cv2.destroyAllWindows()
