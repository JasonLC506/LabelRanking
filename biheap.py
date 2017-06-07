"""
binary heap
March 07, 2017
"""

class BiHeap(object):
    """
    with zero-index, default minimal heap
    supporting tuple heap with key
    """
    def __init__(self):
        self.heap = []  # [[key_value, pointer_originlist]]
        self.length = 0
        self.minial = True
        self.key = None
        self.originlist = {}
        self.maxpointer = None
        self.revmap = {}

    def buildheap(self, list_unsorted, minimal = True, key = None, identifier = None):
        """
        :param list_unsorted:
        :param minimal: True for minimal heap, False for maximal heap
        :param key: key field for comparison
                    for diction or list input, the index can be string or int;
                    default using whole input
        :param identifier: identifier for each item
        """
        self.length = len(list_unsorted)
        self.minimal = minimal
        self.originlist = {pointer: list_unsorted[pointer] for pointer in range(self.length)}
        self.maxpointer = self.length - 1
        self.key = key
        self.identifier = identifier
        self.heap = [[self.itemkeyvalue(self.originlist[index]), index] for index in range(self.length)]
        self.revmap = {self.itemidentifier(self.originlist[index]): index for index in range(self.length)}
        for p in range(self.length/2):
            pindex = self.length/2 - p -1
            self.downsort(pindex)

        ### test
        # assert self.check()
        return self

    def find(self, identifier):
        if identifier in self.revmap.keys():
            return self.revmap[identifier]
        else:
            return None

    def fetch(self, index):
        if index >= self.length:
            return None
        else:
            return self.originlist[self.pointer(self.heap[index])]

    def pop(self, NOTOUT = False):
        item = self.fetch(0)
        if item is not None and not NOTOUT:
            self.delete(0)
        ### test
        # assert self.check()
        return item

    def insert(self, item):
        self.maxpointer += 1
        self.originlist[self.maxpointer] = item
        self.length += 1
        self.heap.append([self.itemkeyvalue(item),self.maxpointer])
        self.revmap[self.itemidentifier(item)] = self.length-1
        self.upsort(self.length - 1)
        ### test
        # assert self.check()
        assert len(self.heap) == len(self.originlist)
        assert len(self.heap) == len(self.revmap)
        return self

    def delete(self, index):
        if index is None:
            return self
        if index >= self.length:
            print self.revmap
            print self.originlist
            print self.heap
            raise ValueError("index out of range")
        temp = self.heap[index]
        enditem = self.heap[self.length -1]
        self.heap[index] = self.heap[self.length - 1]
        self.heap[self.length - 1] = temp
        self.revmap[self.itemidentifier(self.fetch(index))] = index
        del self.revmap[self.itemidentifier(self.fetch(self.length -1))]
        del self.heap[self.length -1]
        del self.originlist[self.pointer(temp)]
        self.length += -1
        if index < self.length:
            if self.compare(self.keyvalue(temp), self.keyvalue(self.heap[index])):
                self.downsort(index)
            else:
                self.upsort(index)

        ### test
        # try:
        #     self.check()
        # except ValueError,e:
        #     print "delete index: ", index, "deleted heap_item: ", temp, "enditem: ", enditem
        #     print self.length
        #     print len(self.heap)
        #     raise e
        assert len(self.heap) == len(self.originlist)
        assert len(self.heap) == len(self.revmap)
        return self

    def update(self, index, item):

        olditem = self.fetch(index)
        if olditem is None:
            raise ValueError("index out of range")
        else:
            self.originlist[self.pointer(self.heap[index])] = item
            id = self.itemidentifier(item)
            oldid = self.itemidentifier(olditem)
            del self.revmap[oldid]
            self.revmap[id] = index
            oldvalue = self.itemkeyvalue(olditem)
            newvalue = self.itemkeyvalue(item)
            self.heap[index][0] = newvalue # heap structure dependent
            if self.compare(newvalue, oldvalue):
                self.upsort(index)
            else:
                self.downsort(index)

        ### test
        # assert self.check()
        assert len(self.heap) == len(self.originlist)
        assert len(self.heap) == len(self.revmap)
        return self

    def check(self, heap_index = None):
        ### for test ###
        ## heap check ##
        if heap_index is None:
            heap_index = 0
        cindex_l = self.childindex(heap_index,lr=False)
        cindex_r = self.childindex(heap_index,lr=True)
        if cindex_l is None:
            return True
        keyvalue = self.keyvalue(self.heap[heap_index])
        keyvalue_l = self.keyvalue(self.heap[cindex_l])
        if cindex_r is not None:
            keyvalue_r = self.keyvalue(self.heap[cindex_r])
        if self.compare(keyvalue, keyvalue_l):
            if cindex_r is None:
                pass
            else:
                if self.compare(keyvalue, keyvalue_r):
                    pass
        else:
            print "p: ", heap_index, keyvalue, "cl: ", cindex_l, keyvalue_l, "cr: ", cindex_r, keyvalue_r
            raise ValueError("unstructured heap")
        ck_l = self.check(cindex_l)
        if cindex_r is not None:
            ck_r = self.check(cindex_r)
            if ck_l and ck_r:
                return True
        else:
            return ck_l

    def itemkeyvalue(self, item):
        if self.key is None:
            return item
        else:
            return item[self.key]

    def itemidentifier(self, item):
        if self.identifier is None:
            return item
        else:
            return item[self.identifier]

    def keyvalue(self, heapitem):
        return heapitem[0] # heap structure dependent

    def pointer(self, heapitem):
        return heapitem[1] # heap structure dependent

    def upsort(self, cindex):
        pindex = self.parentindex(cindex)
        if pindex is None:
            return self
        if not self.compare(self.keyvalue(self.heap[pindex]), self.keyvalue(self.heap[cindex])):
            temp = self.heap[cindex]
            self.heap[cindex] = self.heap[pindex]
            self.heap[pindex] = temp
            self.revmap[self.itemidentifier(self.fetch(cindex))] = cindex
            self.revmap[self.itemidentifier(self.fetch(pindex))] = pindex
            self.upsort(pindex)
        ### test
        # assert self.check()
        assert len(self.heap) == len(self.originlist)
        assert len(self.heap) == len(self.revmap)
        return self

    def downsort(self, pindex):
        lc = self.childindex(pindex, lr = False)
        rc = self.childindex(pindex, lr = True)
        if lc is None:
            return self
        if rc is None:
            optiitem, optiindex = self.heap[lc], lc
        else:
            prefer = self.compare(self.keyvalue(self.heap[lc]), self.keyvalue(self.heap[rc]))
            if prefer:
                optiitem, optiindex = self.heap[lc], lc
            else:
                optiitem, optiindex = self.heap[rc], rc
        preferp = self.compare(self.keyvalue(self.heap[pindex]), self.keyvalue(optiitem))
        if not preferp:
            self.heap[optiindex] = self.heap[pindex]
            self.heap[pindex] = optiitem
            self.revmap[self.itemidentifier(self.fetch(pindex))] = pindex
            self.revmap[self.itemidentifier(self.fetch(optiindex))] = optiindex
            self.downsort(optiindex)
        ### test
        # try:
        #     self.check(pindex)
        # except ValueError, e:
        #     print "pindex: %d, optiindex: %d, optiitem: %s, parentitem: %s" % (pindex, optiindex, str(optiitem), str(self.heap[pindex]))
        #     raise e
        return self

    def compare(self, v1, v2):
        if self.minimal:
            if v1 <= v2:
                return True
            else:
                return False
        else:
            if v1 >= v2:
                return True
            else:
                return False

    def childindex(self, pindex, lr = False):
        """
        find child index
        :param pindex: parent index
        :param lr: False for left, True for right
        :return: child_index
        """
        if lr:
            child_index = (pindex + 1) * 2
        else:
            child_index = pindex * 2 + 1
        if child_index >= self.length:
            return None
        else:
            return child_index

    def parentindex(self, cindex):
        if cindex > 0:
            return int(cindex - 1)/2
        else:
            return None


if __name__ == "__main__":
    # h = BiHeap()
    # h.buildheap([[7,2,3.0,4],[2,1,1.0,1],[4,0,1.0,0],[5,3,4.0,2]], key=2, identifier = 0)
    # h.insert([1,2,9.0,0])
    # h.delete(0)
    # h.update(0,[3,7,5.0,9])
    # print h.originlist
    # print h.heap
    # print h.revmap

    queue = BiHeap()
    queue.buildheap([], minimal=False, key = 1, identifier = 0)
    K = 3
    list = [[1,3],[2,5],[3,2],[4,7],[5,3],[6,1]]
    for item in list:
        queue.insert(item)
        if queue.length>K:
            _ = queue.pop()
    for i in range(K):
        print queue.pop()

    # h2 = BiHeap().buildheap([2,3,4,1,5,7], minimal=False)
    # h2.insert(-1)
    # print h2.originlist
    # print h2.heap
    #
    #
    # print h2.find(-2)