# A function convert the cell's linear index into (z,x,y) index
def IndexFinder(indexOfLine):  # The index should be "minus 4" beforehand
    # if type(indexOfLine)!=int:
    #     print("Wrong input type! Convert it into integer.")
    #     indexOfLine = int(indexOfLine)
    # if not 0<=indexOfLine<25410:
    #     raise ValueError("Input index out of range!")
    zid = indexOfLine // (110 * 11) % 21
    xid = indexOfLine % (110 * 11) // 11
    yid = indexOfLine % (110 * 11) % 11
    return xid, yid, zid


# # A function convert the (x,y) index into the cell's true position
# def TruePosition(xid, yid):
#     if type(xid)!=int or type(yid)!=int:
#         print("Wrong input type!")
#         xid = int(xid)
#         yid = int(yid)
#     if not (0<=xid<110 or 0<=yid<11):
#         raise ValueError("Input index out of range!")
#     xtrue = 41.63 + xid * 5
#     ytrue = -25.0 + yid * 5
#     return (xtrue, ytrue)

# A function convert the cell's true position to (x,y) index-like float number
def PadPosition(xtrue, ytrue):
    # if not (39.13 <= xtrue <= 589.13 or -27.5 <= ytrue <= 27.5):
    #     raise ValueError("Input index out of range!")
    xpad = (xtrue - 41.63) / 5.0  # result range from -0.5 to 109.5
    ypad = (ytrue + 25) / 5.0  # result range from -0.5 to 10.5
    return xpad, ypad
