#!/usr/bin/env python


from thrift.transport import TSocket, TTransport
from thrift.protocol import TBinaryProtocol

#import sys
#sys.path.append('gen-py')

from concrete.communication import *
from concrete.communication.ttypes import *
from concrete.structure.ttypes import *

transportOut = TTransport.TMemoryBuffer()
protocolOut = TBinaryProtocol.TBinaryProtocol(transportOut)

foo = Communication()
foo.text = 'foo bar baz'
sectionSegmentation = SectionSegmentation()
section = Section()
sentenceSegmentation = SentenceSegmentation()
sentence = Sentence()
tokenization = Tokenization()
tokenization.tokenList = [Token(text=t) for t in foo.text.split()]
sentence.tokenizationList = [tokenization]
sentenceSegmentation.sentenceList = [sentence]
section.sentenceSegmentation = [sentenceSegmentation] # TODO typo?
sectionSegmentation.sectionList = [section]
foo.sectionSegmentations = [sectionSegmentation]
foo.write(protocolOut)

bytez = transportOut.getvalue()

transportIn = TTransport.TMemoryBuffer(bytez)
protocolIn = TBinaryProtocol.TBinaryProtocol(transportIn)
newComm = Communication()
newComm.read(protocolIn)

if newComm.sectionSegmentations is not None:
    for sectionSegmentation in newComm.sectionSegmentations:
        if sectionSegmentation.sectionList is not None:
            for section in sectionSegmentation.sectionList:
                if section.sentenceSegmentation is not None:
                    for sentenceSegmentation in section.sentenceSegmentation:
                        if sentenceSegmentation.sentenceList is not None:
                            for sentence in sentenceSegmentation.sentenceList:
                                if sentence.tokenizationList is not None:
                                    for tokenization in sentence.tokenizationList:
                                        if tokenization.tokenList is not None:
                                            for token in tokenization.tokenList:
                                                print token.text
