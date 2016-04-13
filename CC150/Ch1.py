#test one string has all unique characters
def test1_1(test_str):
    for i in range(len(test_str)):
        for j in range(i+1, len(test_str)):
            if (test_str[i] == test_str[j]):
                return False
    return True

test1_1("a b c")
test1_1("abcc")
#What if blank doesn't count



#reverse string
def test1_2(test_str):
    result = ""


#reverse words in a sentence
def reverseWords(s):
    """
    :type s: str
    :rtype: str
    """
    result = ""
    str_list = s.split(" ")
    for i in range(len(str_list)-1, 0, -1):
        print(str_list[i])
        result = result + str_list[i] + " "
    return result + str_list[0]

reverseWords("i am a boy")

#Can not handle " a  b"
#Can not specific "" and "  "
#Better Solution:

def reverseWords(s):
    words = s.split()
    return " ".join(reversed(words))
reverseWords(" I am a boy   ")

#test permutation of two string
def test1_3():
    pass