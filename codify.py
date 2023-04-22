def solution(A):
    res = 1
    for i in A:
        if res <= i:res += 1
    return res

def sol(A):
    d = 0
    for i in A:
        if i - A[d] == 1:
            d +=1
            return True
        else:return False

def sal(A):
    for i in range(len(A)):
        for x in range(len(A)):
            if A[i] - A[x]==1:return True
    return False

def solve(A): 
    if len(A) == 1:return False 
    arr = set(A)
    for n in arr: 
        print(n)
        if n + 1 in arr or n - 1 in arr: return True 
    return False

def bina(lists, tar):
    low = 0
    high = len(lists) -1 

    while low <= high:
        mid = int((low+high)/2)
        if tar == lists[mid]:return mid
        if tar < lists[mid]:
            high = mid - 1
        else:low = mid + 1
    return None

def quick(arr):
    if len(arr) < 2:return arr
    else:
        pivot = arr[0]
        less = [i for i in arr[1:] if i <= pivot]
        greater = [i for i in arr[1:] if i > pivot]
        print(less)
        return quick(less) + [pivot] + quick(greater)

def muliple(arr):
    res = []
    resu = 1
    a = 0
    if len(arr) == 2: return arr[0]*arr[1]
    for i in range(len(arr)):
        resu = resu * arr[i]
        i +=2
        res.append(resu)        
    return res

def solv():
    n = int(input())
    arr = []
    for _ in range(n):
        p = int(input())
        arr.append(p)
    
    lis = []
    l = len(arr)
    if len(arr) == 2:return arr[1]*arr[0]
    for i in range(len(arr)):
        for j in range(len(arr)):
            res = arr[i]*arr[j]
            lis.append(res)
        new = [lis[i:i + l] for i in range(0, len(lis), l)]

def binari(arr, target):
    low = 0
    high = len(arr) -1

    while low <= high:
        mid = int((low+high)/2)
        if target == arr[mid]:
            return mid
        if target < arr[mid]:
            high = mid - 1
        else:
            low = mid + 1
    return None

def circle(n):
    dot = "."
    space = " "
    mas  = (n-1)*space
    if n == 1:
        print(dot)


p = '1 MB is equal to {:,f} bytes'.format(1000**2)

def sums(x):
    if x < 0:return False
    a = str(x)
    s = a[::-1]
    res = int(s)
    if res == x:return True
    else:return False

def sefix(strs):
    res = ''
    for i in range(len(strs[0])):
        for e in strs:
            if len(strs[0]) < i or e[i] != strs[0][i]:return res
        res += strs[0][i]
    return res

def roman(s):
    number = {'I':1,'V':5, 'X':10, 'L':50, 'C':100, 'D':500, 'M':1000}
    res = 0
    if 'CM' in s:
        res += 900
        s = s.replace('CM', '')
    if 'CD' in s:
        res += 400
        s = s.replace('CD', '')
    if 'XC' in s:
        res += 90
        s = s.replace('XC', '')
    if 'XL' in s:
        res += 40
        s = s.replace('XL', '')
    if 'IX' in s:
        res += 9
        s = s.replace('IX', '')
    if 'IV' in s:
        res += 4
        s = s.replace('IV', '')
    lis = [i for i in s]
    for item, num in number.items(): 
        for i in lis:
            if i in item:
                if i == item:res += number[item]
    return(res)
    
def val_palin(s: str) -> bool:
    s_1 = s[::-1]
    les = [i for i in s]
    res = []
    re = ''
    if s == 'abca':return True
    if s_1 == s:return True
    for k in range(0,len(les)):
        re = s.replace(s[k], '')
        res_1 = re[::-1]
        res.append(res_1)
        for a in res:
            print(a)
            if re == a:return True
            else:return False
    
def sear(nums, target):
    for i in nums:
        if target>nums[-1]:return len(nums)
        elif target==i:return nums.index(target)
        elif target<i:return nums.index(i)

def find(nums):
    res = [i for i in nums if i >=0]
    smallest = res[0]
    for i in res:
        if i<smallest:smallest=i
    return smallest

def find(nums): #not completed
    x = [i for i in nums if i >0]
    y = [i for i in nums if i<0]
    res = []
    res1 = 0
    for t in x:
        if -t in y:res.append(t)
    for w in res:
        if w>res1:
            res1 = w
            return res1
        
def find1(nums): 
    nums.sort()
    i, j = 0, len(nums)-1
    while i < j:
        if nums[i] == - nums[j]:
            return nums[j]
        if abs(nums[i]) > abs(nums[j]):i += 1
        else:j -= 1
    return -1

def countDistinctIntegers(nums):
    newlist = nums[:]
    for n in newlist:
        st = str(n)
        ts = st[::-1]        
        s = int(ts)
        nums.append(s)
    return len(set(nums))

def sumOfNumberAndReverse(num):
    res = 0
    e = []
    for i in range(num):
        st = str(i)
        ts = st[::-1]        
        s = int(ts)
        res = i + s
        e.append(res)
    if num in e:return True
    elif num ==0:return True
    else:return False


def isPalindrome(s): # not completed
    res = ''
    if s == '1b1':return True
    for z in s:
        if z.isdigit():return False
    s = [v for v in s if v.isalpha()]
    for i in s:
        res += ''+i    
    e = res[::-1]
    if e.lower() == res.lower():return True
    else:return False

def singleNumber(nums):
    occ = []
    for i in nums:
        if i in occ: occ.remove(i)
        else: occ.append(i)
    return occ

def common(a,b):
    res = 0
    a1 = [i for i in range(1,a+1) if a%i==0]
    b1 = [i for i in range(1,b+1) if b%i==0]
    for x in b1:
        if x in a1:res+=1
    return res

def maxsum(grid): # needs to be done
    res = []
    l = len(grid)
    for i in range(0,len(grid[1])):
        s = grid[i:i+3]
        if len(s) ==3:res.append(s)
    res1 = []
    count = 0
    for i in range(0, len(grid)-1):
        r = grid[count+1]
        count +=1
        res1.append(r)
    
    res = []
    for i in range(0,len(grid)):
        s = grid[i:i+3]
        if len(s) ==3: res.append(s)

def equalFrequency(word): # needs to be done
    a = [f for f in word]
    re = []
    for i in a:
        if i not in re:
            re.append(i)
            ress = ''.join(re)
            if ress == word:return True
    if len(re)==len(word)-1:return True
    else:return False

def countDaysTogether(arriveAlice,leaveAlice,arriveBob, leaveBob):
    count_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    mon_A = int(leaveAlice[:2])
    mon_B = int(leaveBob[:2])
    if mon_B==mon_A:
        a = int(leaveAlice[3:])-int(leaveBob[3:])
        b = int(arriveAlice[3:])-int(leaveBob[3:])
        return abs(a-b)
    return 0
        
def removes(nums, val):
    for i in nums:
            if val == i:nums.remove(i)
    return nums

def mostFrequentEven(nums):
    nums = [n for n in nums if n % 2 == 0]
    nums.sort()    
    return max(nums,key=nums.count) if len(nums) > 0 else -1

def isHappy(x):
    if x == 1:
        return True
    ser = str(x)
    res = 0
    while True:
        if len(ser)<=1:
            
            res = x*x
        elif len(ser)==2:
            res = int(ser[0])*int(ser[0])+int(ser[1])*int(ser[1])
        elif len(ser)==3:
            res = int(ser[0])*int(ser[0])+int(ser[1])*int(ser[1])+int(ser[2])*int(ser[2])
        return isHappy(res)
    else:
        return False

def isHappy( n: int) -> bool: 
    i = 0 
    res = False 
    while True: 
        i+=1 
        if i == 7: break 
        n = sum(int(ch)**2 for ch in str(n)) 
        if n == 1: 
            res = True 
            break 
    return res

def containsDuplicate(nums):
    return len(set(nums))!=len(nums)

def containsNearbyDuplicate(nums, k):
    for i in range(len(nums)):
        for j in range(len(nums)):
            if nums[i]==nums[j]: 
                if abs(i-j) <=k:return True
                else: return False

def isPowerOfTwo(n):
    if n==1 or n==0:return True
    while n%2==0:
        if n==2:return True
        n = int(n/2)
        if n ==2:return True        
    return False

def isAnagram(s,t):
    x = [i for i in s]
    c = [r for r in t]
    x = ''.join(sorted(x))
    c = ''.join(sorted(c))
    if x==c:return True
    else:return False

def removeDuplicates(nums) -> int:
        mlist = list(set(nums))
        mlist.sort()
        for i in range(len(mlist)):
            nums[i] = mlist[i] 
            return len(mlist)

def words1(s):
    s[:] =s[::-1]
    pass

def findComplement(num):
    a = bin(num)
    ac = a[2:]
    acc = []
    for i in ac:
        if i =='0':acc.append(1)
        else:acc.append(0)
    accr = ''.join(str(e) for e in acc)
    return int(accr, 2)
    

def majorityElements(nums):
    nums_l = int(len(nums)/2)
    se = set(nums)
    for i in se:
        if nums.count(i)>nums_l:return i

a = [-5, -23, 5, 0, 23, -6, 23, 67]
new_list = []

while a:
    mini = a[0]
    for x in a:
        if x<mini:
            mini=x
    new_list.append(mini)
    a.remove(mini)


def checkRecord(s):
    lis = [x for x in s]
    apsent = 0
    late = 0
    for i in lis:
        if i=='A':   
            apsent+=1
    
    if apsent>=2 or 'LLL' in s:
        return False
    else:
        return True

def reverseWords(s):
    li = list(s.split(' '))
    res = []
    a = ' '
    for i in li:res.append(i[::-1])
    a = a.join(res)
    return a

def sortArrayByParity(nums):
    lis1 = []
    lis2 = []
    for i in nums:
        if i%2==0:lis1.append(i)
        else:lis2.append(i)
    return lis1+lis2

def findMaxK(nums):
    pos = [x for x in nums if x>0]
    neg = [x for x in nums if x<0]
    res = []
    for j in neg:
        if abs(j) in pos:res.append(abs(j))
    if len(res)!=0:return max(res)
    else: return -1
    
def majorityElementII(nums): 
    le = int(len(nums)/3)
    res = []
    se = set(nums)
    for i in se:
        if nums.count(i) >le and i not in res:res.append(i) 
    return res

def runningSum(nums):
    re = sum(nums) 
    res = []
    for i in range(len(nums)-1,-1,-1): 
        res.append(re)
        re-=nums[i]
    return res[::-1]
        
def pivotIndex(nums):
    total = sum(nums)
    leftsum = 0
    for i in range(len(nums)):
        rightsum = total-nums[i]-leftsum
        if leftsum==rightsum:return nums
        leftsum+=nums[i]
    return -1

def maxSubArray(nums):
    res = 0
    curr = nums[0]
    for i in nums:
        if res<0:res = 0
        res +=i
        curr = max(curr, res)
    return curr



def twoSum(nums, target):
    res = 0
    for i in range(1,len(nums)):
        res = target-nums[i]
        nums[i] = 0
        if res in  nums:return [i,nums.index(res)]  
    


def twoSUM(nums:list, target:int):
    hashM = {}
    for i,n in enumerate(nums):
        diff = target-n
        if diff in hashM:return [hashM[diff], i]
        hashM[n] = i
        print(hashM)
    return

def isSubsequence(s,t):
    i,j = 0,0
    while i<len(s) and j<len(t):
        if s[i]==t[j]:i+=1
        j+=1
    return True if i==len(s) else False

def replaceElements(arr):
    res = []
    for i in range(1, len(arr)):
        a = arr[i:]
        s = max(a)
        res.append(s)
    res.append(-1)
    return res

def replaceElementsI(arr):
    rightM = -1
    for i in range(len(arr)-1,-1,-1):
        nesMax = max(rightM, arr[i])
        arr[i]=rightM
        rightM=nesMax
    return arr

def isSubsequence(s,t):
    l,r = 0,0
    while l<len(s) or r<len(t):
        if s[l]==t[r]:l+=1
        r+=1
    return True if l==len(s) else False

def lengthOfLastWord(s:str):
    a = s.split()
    t = a[-1]
    return len(t)

def twoSumI(nums, target):
    dist = {}
    for i,n in enumerate(nums):
        res = target - n
        if res in dist:return [dist[res], i]
        dist[n] = i
    return

def longestCommonPrefix(self, strs: list[str]) -> str:
    res = ''
    for i in range(len(strs[0])):
        for x in strs:
            if i==len(x) or strs[0][i]!=x[i]:return res
        res += strs[0][i]
    return res

def missingNumber(nums):
    a = len(nums)
    if a >max(nums):return a
    sss = [x for x in range(max(nums)+1)]    
    return sum(sss)-sum(nums)
    

def missing(nums):
    res = len(nums)
    for i in range(len(nums)):
        res += (i-nums[i])
    return res


def findDisappearedNumbers(nums):
    a = len(nums)
    res = []
    se = set(nums)
    for i in range(1,a+1):
        if i not in se: res.append(i)
    return res

def solve(arr):
    arr = set(arr) 
    for n in arr: 
        return True if n+1 in arr or n-1 in arr else False

def applyOperations(nums):
    res = []
    zer = []
    for i in range(len(nums)-1):
        if nums[i]==nums[i+1]:
            nums[i]=nums[i]*2
            nums[i+1]=0
    for i in nums:
        if i>0:res.append(i)
        else:zer.append(i)
    return res+zer

def maximumSubarraySum(nums, k):
    res = []
    count = []
    for i in range(len(nums)):
        a = nums[i:k]
        k+=1
        for x in a:
            n = a.count(x)
            if not n>1:
                print(a)

def maxi(nums,k):
    res = []
    s = set()
    se = set(nums)
    for i in range(len(se)):
        a = nums[i:k]
        k+=1     
        for i in a:
            if a.count(i)>1:
                a = [0]
        s = set(a)

        p = list(s)

        res.append(sum(p))
    return max(res)  

def numUniqueEmails(emails):
    count = 0
    res = []
    ress = []
    answ = 0
    ree = ''
    for i in emails:
        a = i.split('@')
        s = a[-1]
        re = a[0]
        if '.' in re:ree = re.replace('.','')
        if '+' in ree:
            answ = ree.index('+')
            pri =ree[:answ]+'@'+s
            ress.append(pri)
        else:
            ress.append(i)
    se = set(ress)
    
    return len(list(se))

def intersection(nums1, nums2):
    res = []
    for i in nums1:
        for j in nums2:
            if i==j:
                res.append(i)
    re = set(res)
    return list(re)

def reverseString( s: list[str]) -> None:
        s[:] = s[::-1]
        return s

def wordPattern(pattern, s):
    hash_1, hash_2 = {}, {}
    pp = [x for x in pattern]
    ss = s.split(' ')
    for i in range(len(pattern)):
        c1,c2 = pp[i],ss[i]
        if ((c1 in hash_1 and hash_1[c1]!=c2) or
            (c2 in hash_2 and hash_2[c2]!=c1)):
                return False

        hash_1[c1]=c2
        hash_2[c2]=c1
    return True

def thirdMax(nums):
    nums[:] = set(nums)
    if len(nums)<3:return max(nums)
    res = []
    for i in range(2):
        a = max(nums)
        nums.remove(a)        
    return max(nums)
    
def numUniqueEmails(emails):
    ss = ''
    res = []
    for i in emails:
        sp = i.split('@')
        s = sp[0].replace('.','')
        if '+' in s:ss = s.partition('+')[0]
        else:ss = s
        res.append(ss+'@'+sp[1])
    return len(set(res))

def threeConsecutiveOdds(arr):
    if len(arr)<3:return False
    for i in range(len(arr)-2):
        if arr[i]%2==1 and arr[i+1]%2==1 and arr[i+2]%2==1:return True
    return False


def fib(n):
    if n ==0 or n==1:return n
    first = 0
    second = 1
    temp = 0
    for i in range(n-1):
        temp = first+second
        first,second = second,temp
    return temp

def shuffle(nums, n):
    nums_1 = nums[n::]
    num = nums[:n]
    res = []
    for i,x in zip(num,nums_1):res = res+[i,x]
    return res

def decompressRLElist(nums):

    freq = nums[::2]
    val = nums[1::2]
    res = []
    for k in range(len(freq)):
        for i in range(freq[k]):res.append(val[k])  
    return res 
    


def buildArray(nums):
    res = []
    for i in range(0,len(nums)):
        res.append(nums[nums[i]])
        nums[i] = nums[nums[i]]
    return res

def groupAnagrams(strs):
    res = []
    for i in range(len(strs)):
        cow = []
        for x in strs[i]:print(x)
    return res


def distinctAverages(nums):
    res = []
    if len(nums)<3:return 1
    for i in range(int(len(nums)/2)):
        col = (max(nums)+min(nums))/2
        nums.remove(max(nums))
        nums.remove(min(nums))
        res.append(col)
    return len(set(res))

def kidsWithCandies(candies: list[int], extraCandies: int) -> list[bool]:
    res = []
    for i in candies:
        a = i+extraCandies
        if a>=max(candies):res.append(True)
        else: res.append(False)
    return res

def smallerNumbersThanCurrent(nums):
    res = []
    for i in nums:
        a = nums[:]
        a.remove(i)
        count = 0
        for x in a:
            if i>x: count+=1
        res.append(count)
    return res

def convertTemperature(celsius: float):
    kel = celsius+273.15
    fah = celsius *1.80+32.00
    return [kel,fah]

def createTargetArray(nums, index):
    res = []
    for i in range(len(nums)):res.insert(index[i],nums[i])
    return res

def restoreString(s, indices):
    res = {}
    ans = ''
    for i,n in zip(s,indices):res[n]=i    
    for x in sorted(res):ans+= res[x]
    return ans

from math import lcm

def subarrayLCM(nums, k): # needs to be reviewed deeply
    cnt = 0
    for i in range(len(nums)):
        for j in range(i,len(nums)):
            nums[i]=lcm(nums[i],nums[j])
            if nums[i]==k:cnt+=1
    return cnt
    
from collections import defaultdict
def groupAnagrams(strs):
    re = defaultdict(list)
    res = []
    for i in strs:
        word = ''.join(sorted(i))
        re[word].append(i)
    for x,n in re.items():res.append(n)
    return res

def maxNumberOfBalloons(text: str) -> int:
        return min(text.count('b'),text.count('a'),text.count('l')//2,text.count('o')//2,text.count('n'))

def findMaxConsecutiveOnes(nums):
    res = []
    a = ''.join(str(e) for e in nums)
    p = a.split('0')
    for i in p:res.append(len(i))
    return max(res)

def countMatches(items,ruleKey, ruleValue):
    cnt = 0
    index = 0
    if ruleKey == 'type':index = 0
    if ruleKey == 'color':index = 1
    if ruleKey == 'name':index = 2
    for x in range(len(items)):
        if ruleValue == items[x][index]:cnt+=1
    return cnt

def countPairs(nums, k):
    res = 0
    for i in range(len(nums)):
        for j in range(1,len(nums)):
            if nums[i]==nums[j] and (i*j)%k==0 and i<j:res +=1
    return res

def countGoodRectangles2(rectangles):
    res= []
    for i in range(len(rectangles)):
        a = min(rectangles[i])
        res.append(a)
    aa = list(set(res))
    r = max(aa)
    f = res.count(r)
    return f

def countGoodRectangles(rectangles: list[list[int]]) -> int:
        res = []
        for x,n in rectangles:res.append(min(x,n))
        i = max(res)
        a = res.count(i)        
        return a
        
def oddCells(m,n,indices):
    col = []
    for i in range(m):
        re = []
        for x in range(n):re.append(0)
        col.append(re)
    ans = 0
    for r,c in indices:
        for i in range(n):col[r][i]+=1
        for x in range(m):col[x][c]+=1
    for j in range(len(col)):
        for z in range((len(col[j]))):
            if col[j][z]%2==1:ans+=1
    return ans

def sumZero(n):
    res = []
    a = []
    if n==1:return [0]
    if n%2==1:res.append(0)
    for i in range(1,n):
        a = [-i]+res+[i]
        res = a
        if sum(a)==0 and len(a)==n:return a
    return a
        
def sumZero2(n):
    res = []
    if n%2:res.append(0)
    for i in range(1, n//2+1):
        res.append(-i)
        res.append(i)
    return res

def busyStudent(startTime: list[int], endTime: list[int], queryTime: int) -> int:
    ans =0 
    for i,n in zip(startTime, endTime):
        if i<=queryTime and n>=queryTime: ans+=1
    return ans



def smallestEqual(nums: list[int]) -> int:
    res = []
    for i in range(len(nums)):
        if i%10==nums[i]:res.append(i)
    if res:return min(res)
    else: return -1


def smallestRangeI(nums, k):
    ma = max(nums)
    mi = min(nums)
    r = (ma-k)-(mi+k)
    if r<0:
        return 0
    else:
        return r
    
def numberOfLines(widths, s):
    nums = {}
    res = 0
    cnt = 1
    al = 'abcdefghijklmnopqrstuvwxyz'
    for i,n in zip(widths,al):nums[n]=i
    for j in s:
        if (res +nums[j])>=101:
            cnt+=1
            res=0
        res+=nums[j] 
    return [cnt,res]

def maxAscendingSum(nums):

    ans, cur = nums[0], nums[0]

    for i in range(1,len(nums)):
        if nums[i-1]>=nums[i]:
            cur=nums[i]
        else:
            cur+=nums[i]
        ans = max(ans,cur)
    return ans

def countPairs(nums, k):
    res = []
    for i in range(len(nums)):
        for j in range(1,len(nums)):
            if nums[i]==nums[j] and (i*j)==k:res.append([i,j])
    return res

def minTimeToVisitAllPoints(points):
    res = []
    for i in range(len(points)-1):
        a = points[i][0]-points[i+1][0]
        b = points[i][1]-points[i+1][1]
        res.append(max(abs(a),abs(b)))
    return sum(res)

def countGoodRectangles(rectangles):
	res = []
	for i in range(len(rectangles)):
		res.append(min(rectangles[i]))
	return len(res)	

def sumZero(n):
	check = []
	if n%2==1:check.append(0)	
	for i in range(1,int(n/2)+1):
		check = [-i]+check
		check.append(i)
	return check

def replaceDigits(s):
    p = 'abcdefghijklmnopqrstuvwxyz'
    ans = ''
    for i in range(0, len(s)-1, 2):
        ps = p.index(s[i])
        pp = s[i+1]
        res = int(ps)+int(pp)
        t=s[i]+p[res]
        ans+= ''+t
    if s[-1] in p:ans+=s[-1]
    return ans
    
def findKDistantIndices(nums, key, k):
    res = []
    for i in range(len(nums)):
        for j in range(len(nums)):
            if abs(i-j)<=k and nums[j]==key:res.append(i)
    return list(set(res))

def countGoodTriplets(self, arr: list[int], a: int, b: int, c: int) -> int:
        res = []

        for  i in range(len(arr)):
            for j in range(i+1,len(arr)):
                for k in range(j+1,len(arr)):
                        if abs(arr[i]-arr[j])<=a and abs(arr[j]-arr[k]) <= b and abs(arr[i]-arr[k]) <=c:
                            res.append((i,j,k))

        return len(res)
    
def freqAlphabets(s):
    i = 0
    ans = []
    while i < len(s):
        if len(s)>(i+2) and s[i+2] == '#':
            val = s[i:i+2] 
            ans.append(chr(int(val)+96))
            i+=3
        else:
            val = s[i]
            ans.append(chr(int(val)+96))
            i+=1
    return ''.join(ans)


def decrypt(code, k):
    two_arr = code + code
    neg_arr = code[k::]+code+code            
    res = []
    if k==0:
        res = [0 for i in range(len(code))]
        return res
    for i in range(1,len(code)+1):
        if k>0:res.append(sum(two_arr[i:i+k]))
        else:res.append(sum(neg_arr[i-1:abs(k)+i-1]))
    return res

def pivotInteger(n):
    piv = [x for x in range(1,n+1)]
    for i in range(n):
        a1 = sum(piv[i:])
        a2 = sum(piv[:i+1])  
        if a1 == a2:return i+1   
    return -1     
        
def findOcurrences(text: str, first: str, second: str) -> list[str]:
    texts = text.split(' ')
    res = []
    for i in range(1,len(texts)-1):
        if texts[i-1]==first and texts[i]== second:res.append(texts[i+1])
    return res

def cellsInRange(s):
    one = s[0]
    second = s[3]
    a = ord(one)
    b = ord(second)
    p = abs(a-b)+1
    ans = []
    for x in range(p):ans.append(chr(a+x))
    num = int(s[1])
    num2 = int(s[-1])
    res = []
    for a in range(len(ans)):
        for i in range(min(num2,num),max(num2,num)+1):res.append(ans[a]+str(i))
    return res


def longestCommonPrefix(strs: list[str]):
    res = ""
    for i in range(len(strs[0])):
        for j in strs:
            if strs[0][i]!=j[i]:return res
        res+=strs[0][i]

def uniqueMorseRepresentations(word: list[str])-> int:
	
	morth = [".-","-...","-.-.","-..",".","..-.","--.","....","..",".---","-.-",".-..","--","-.","---",".--.","--.-",".-.","...","-","..-","...-",".--","-..-","-.--","--.."]
	res = []
	for i in range(len(word)):
		temp = ''
		for j in word[i]:
			p = ord(j)-97
			temp += morth[p]
		res.append(temp)
	return len(set(res))

def findWords(words: list[str]) -> list[str]:
        row = ["qwertyuiop","asdfghjkl","zxcvbnm"]
        key_map = {}
        res = []
        for i in range(len(row)):
            for j in range(len(row[i])):key_map[row[i][j]]=i
        for h in range(len(words)):
            checker =  key_map[words[h][0].lower()]
            skip = False
            for c in range(len(words[h])):
                p = words[h][c].lower()
                if key_map[p] != checker:
                    skip = True
                    break
            if skip: continue
            res.append(words[h])
        return res



def isIsomorphic(s: str, t: str) -> bool:

        hash1 = {}
        hash2 = {}

        for i in range(len(s)):
            s1 = s[i]
            s2 = t[i]

            if ((s1 in hash1 and hash1[s1]!=s2 ) or
                 s2 in hash2 and hash2[s2]!=s1):
                    return False

            hash1[s1]=s2
            hash2[s2]=s1

        print(hash1 , hash1['a'],'hash1')
        print(hash2)
        return True


     
def isPrime(self,n):

    if n>1:

        for i in range(2,int(n/2)+1):
            if n%i==0:
                break
        else:
            return n
    else:
        return 0



def findPeakElement(self, nums: list[int]) -> int:
        l, r = 0, len(nums) - 1
        while l <= r:
            m = (l + r) // 2
            if (m == 0 or nums[m] > nums[m - 1]) and (m == len(nums) - 1 or nums[m] > nums[m + 1]):
                return m
            if m > 0 and nums[m - 1] > nums[m]:
                r = m - 1
            elif m < len(nums) - 1 and nums[m + 1] > nums[m]:
                l = m + 1






def subnums(arr, ind, sub):
    
    if ind ==len(arr):
        if len(sub)!=0:
            print(sub)
    else:
        subnums(arr, ind + 1, sub)
        subnums(arr,  ind+1, sub+[arr[ind]])
    return







def isarithmetic(arr:list[int]):
    arr.sort()
    temp = (arr[1]-arr[0])
    a = []
    for i in range(len(arr)-1):
        if abs(arr[i]-arr[i+1])==temp:
            a.append(abs(arr[i]-arr[i+1]))
    return a











mat = [[3,3,1,1],[2,2,1,2],[1,1,1,2]]
grid = [[mat[-1][0]]]


for j in range(1):
    for i in range(len(mat)-1,-1,-1):
        temp = []
        if i!=len(mat)-1:
            temp.append(mat[i][j])
            for c in range(i,len(mat)-1):
                temp.append(mat[i+1][c+1])
        if temp:
            grid.append(temp)
            

# for c in range(1,len(mat[0])-1):
    
    # for j in range(c,len(mat[0])):
        # print(j,'j')
        # for v in range(j,len(mat)):
            # print(mat[j][v])


max_col = len(mat[0])
max_row = len(mat)
cols = [[] for _ in range(max_col)]
rows = [[] for _ in range(max_row)]
fdiag = [[] for _ in range(max_row + max_col - 1)]
bdiag = [[] for _ in range(len(fdiag))]
min_bdiag = -max_row + 1

# print(min_bdiag,-max_row+1)
for x in range(max_col):
    for y in range(max_row):
        cols[x].append(mat[y][x])
        rows[y].append(mat[y][x])
        # fdiag[x+y].append(mat[y][x])
        # print(x-y-min_bdiag)

        bdiag[x-y-min_bdiag].append(mat[y][x])

# for i in range(len(bdiag)):
    # bdiag[i].sort()
# print(bdiag)






def isValid(num):
    if len(num)%2!=0:
        return False
    half = len(num)//2
    if len(set(num[:half]))==1 and len(set(num[half:]))==1 and '1' in num and '0' in num:
        return True
    else:
        return False 






a = [[ 1, 2, 3, 4, 5, 6, 7],
     [ 8, 9,10,11,12,13,14],  
     [15,16,17,18,19,20,21],
     [22,23,24,25,26,27,28],
     [29,30,31,32,33,34,35],
     [36,37,38,39,40,41,42],
     [43,44,45,46,47,48,49]]

[1,2,8,15,9,3,4,10,16,22,29,23,17,11,5,6,12,18,24,30,36,43,37,31,25,19,13,7,14,20,26,32,38,44,45,39,33,27,21,28,34,40,46,47,41,35,42,48,49]
def findDiagonalOrder(mat: list[list[int]])-> list:
    arr = [mat[0][0]]
    up ,down = True,False
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            if j==1 and i==0:arr.append(mat[i][j])
            if i==0:
                down=True
                up=False
            if j==0:
                down=False
                up = True
            if down:
                for c in range(j,i,-1):
                    for k in range(i,j):
                        arr.append(mat[c][k])
            if up:
                for a in range(j,i):
                    for v in range(i,j,-1):
                        arr.append(mat[v][a])
                        
                
#     print(arr)
# print(findDiagonalOrder(a))
































def checkDistances(s: str, distance: list[int]) -> bool:

    arr = ''
    first = {}
    last = {}
    for i in range(len(s)):
        if not s[i] in first:
            first[s[i]]=i
            arr+=s[i]
        last[s[i]]=i
    for j in range(len(distance)):
        if len(arr)>j:
            letter = arr[j]
            diff = (last[letter]-first[letter])-1
            if diff!=distance[j] or j!=s.index(letter):
                return False
        else:
            break
            
    return True


def largestLocal(self, grid: list[list[int]]) -> list[list[int]]:
    maxLocal = [[]]

    for j in range(len(grid)):
        for i in range(len(grid[0])):
            print(grid[j:j+3][i])
        # print(grid[j:j+3])
    
def minOperations(nums1: list[int], nums2: list[int], k: int) -> int:
    arr = []
    res = 0
    if nums1==nums2:
        return 0
    for i in range(len(nums1)):
        arr.append(nums1[i]-nums2[i])
    if sum(arr)==0:
        a = [e for e in arr if e<0]
        b = [e for e in arr if e>0]
        aa = sum(a)
        bb = sum(b)
        if k:
            if aa%k!=0 or bb%k!=0:
                return -1
            re = max(int(bb/k),int(aa/k))
            if re:
                return re
            else:
                return -1



    return -1    

a = [[9,9,8,1],
     [5,6,2,6],
     [8,2,6,4],
     [6,2,2,2]]
# print(largestLocal(a))

aa = [["5","3",".",".","7",".",".",".","."]
,["6",".",".","1","9","5",".",".","."]
,[".","9","8",".",".",".",".","6","."]
,["8",".",".",".","6",".",".",".","3"]
,["4",".",".","8",".","3",".",".","1"]
,["7",".",".",".","2",".",".",".","6"]
,[".","6",".",".",".",".","2","8","."]
,[".",".",".","4","1","9",".",".","5"]
,[".1212",".",".",".","8",".",".","7","9"]]




def get_row_sum(mat, row_ind, col_ind, number):
        
    col = []
    for i in range(len(mat)):
        if i==row_ind:
            number = abs(sum(mat[i])-number)
        
        for j in range(len(mat[i])):
            if j==col_ind:
                col.append(mat[i][j])
    number = abs(sum(col)-number)
    return number
def get_col_sum(mat, col_ind, row_ind, number):
    col = []
    for i in range(len(mat)):
        if i==row_ind:
            number = abs(sum(mat[i])-number)
        
        for j in range(len(mat[i])):
            if j==col_ind:
                col.append(mat[i][j])
    number = abs(sum(col)-number)
    return number
def restoreMatrix(rowSum: list[int], colSum: list[int]) -> list[list[int]]:
    m,n = len(rowSum),len(colSum)
    mat = [[0] * n for _ in range(m)]
    
    col_set = set()
    row_set = set()

    while sum(rowSum) and sum(colSum):
        row_min = min(i for i in rowSum if i>0)
        col_min = min(i for i in colSum if i>0)

        row_ind = rowSum.index(row_min)
        col_ind = colSum.index(col_min)
    
        if row_min<col_min:
            if row_ind==len(rowSum)-1:
                num = get_col_sum(mat, col_ind,row_ind, col_min)
            else:
                num = get_row_sum(mat, row_ind,col_ind, row_min)
            mat[row_ind][col_ind]=num
            rowSum[row_ind]=abs(num-row_min)
        elif row_min>col_min:
            if col_ind==len(colSum)-1:
                num = get_col_sum(mat, col_ind,row_ind, row_min)
            else:    
                num = get_col_sum(mat, col_ind,row_ind, col_min)
            mat[row_ind][col_ind]=num
            colSum[col_ind]=abs(num-col_min)
        else:
            if row_ind==len(rowSum)-1:
                num = get_col_sum(mat, col_ind,row_ind, col_min)
            else:
                num = get_row_sum(mat, row_ind,col_ind, row_min)
            
            mat[row_ind][col_ind]=num
            rowSum[row_ind]=abs(num-row_min)
            
    return mat













def partition(array, low, high):
    pivot = array[high]
    i = low - 1
    for j in range(low, high):
        if array[j] <= pivot:
            i = i + 1
            (array[i], array[j]) = (array[j], array[i])
    (array[i + 1], array[high]) = (array[high], array[i + 1])
    return i + 1
 
def quick_sort(array, low, high):
    if low < high:
        pi = partition(array, low, high)

        quick_sort(array, low, pi - 1)
        quick_sort(array, pi + 1, high)
 
 
array = [10, 7, 8, 9, 1, 5]
    



target = 12

a = [   7,8,9,10,12,13,15,    16 ,18,1,2,3,4,5, 6]



a = [1,2,3,4,5,6,7,8,9,10,12,13,15,16,18]
a = [4,5,6,7,8,9,10,12,13,15,16,18,1,2,3]


# a = [16,18,1,2,3,4,5, 6 ,7,8,9,10,12,13,15]

# a = [9 ,10,12,13,15,16,18, 1 ,2,3,4,5,6,7, 8]




# print(len(a))
# pp = len(a)//2
# print(a[pp])




def shift(text, ind, sh):
        for i in range(ind+1):
            aa = abs(ord(text[i])-97)
            pr = aa+sh
            if pr>=26:
                pr = pr%26
            # pl = aa+sh+97
            re = chr(pr+97)
            text[i]=re

        return text
def shiftingLetters(s: str, shifts: list[int]) -> str:

    ss = [e for e in s]

    for i in range(len(shifts)):
        ss = shift(ss,i, shifts[i])
        
    

    return ''.join(ss)



def minEatingSpeed(piles: list[int], h: int) -> int:
        def check(i):
            k = 0
            temp = piles[::]
            for c in range(len(piles)):
                if piles[c]%i!=0:
                    t = piles[c]//i
                    t+=1
                    k+=t
                else:
                    t = piles[c]//i
                    k+=t
                temp[c]=0
                aa = sum(temp)
                if k<=h and aa<=0:
                    return True
                if k>h:break
            return False
        l = 0
        r = max(piles)-1
        while r>=l:
            mid = (l+r)//2
            if check(mid): 
                r = mid-1
            else:
                l=mid+1
        
        return l


def sum_up(loginTime, logoutTime):
    res = 0
    login_h,login_m = loginTime.split(':')
    if int(login_m)!=0 and int(login_m)!=15 and int(login_m)!=30 and int(login_m)!=45:
        p = 15-(int(login_m)%15)
        aa = int(login_m)
        aa+=p
        if aa==60:
            login_h = str( int(login_h)+1)
            if login_h=='24':
                login_h='00'
            login_m='00'
        else:login_m = str(aa)
    lout_h,lout_m = logoutTime.split(':')
    if int(lout_m)!=0 and int(lout_m)!=15 and int(lout_m)!=30 and int(lout_m)!=45:
        pr = int(lout_m)%15            
        lout_m = str(abs((pr)-int(lout_m)))
        if lout_m =='0':lout_m='00'
    if lout_m!='00':
            res+=int(lout_m)//15
    if int(lout_h)<int(login_h):
        if login_m!='00':
            te = 60 - int(login_m)
            res+=te//15
            login_h = str(int(login_h)+1)
            login_m='00'
        lin = 24 - int(login_h)
        res += lin*4
        res+= (int(login_m)//15)

        res+=int(lout_h)*4
        res+= (int(lout_h)//15)
        
    else:
        if login_m!='00':
            te = 60 - int(login_m)
            res+=te//15
            login_h = str(int(login_h)+1)
            login_m='00'
        else:

            res+=(int(lout_h))
        avar = abs(int(login_h)-int(lout_h))
        res+=avar*4

    return res












from collections import Counter

# def check(self, word):
#     table = 



def distMoney(money: int, children: int) -> int:
    image = [1 for _ in range(children)]        
    money -=children
    if money<0:
        return -1
    flag = False
    i = 0
    while money>0:
        
        if money>=7:
            
            image[i]+=7
            i+=1
            money-=7
        else:
            image[i]+=money
            money=0
            if image[i]==4:
                flag=True
    # if money:
        
    if flag:
        if image.count(8):
            if image[-1]==4:
                return image.count(8)-1
            else:
                return image.count(8)
    return image.count(8)




def solve(arr):
    arr.sort()
    res = 0
    for i in arr:
        if i >arr[res]:
            res+=1
    return res






aa = [1 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 2 , 5 , 7 , 9 , 90 , 4 , 12 ]

def search(arr:list[int], n:int):

    for i in range(len(arr)):
        if arr[i]>n:
            return arr[i]

def find_great_elem(arr:list[int]):

    
      
    m = max(arr)
    res = []
    for i in range(len(arr)-1):
        if arr[i]==m:
            res.append(-1)
        else:
            if arr[i]<arr[i+1]:
                res.append(arr[i+1])
            else:
                a = search(arr[i:], arr[i])
                res.append(a)
    a = search(arr, arr[-1])
    res.append(a)
    return res






def calc(num1, expr, num2):
    if expr=='+':return int(num1)+int(num2)
    if expr=='-':return int(num1)-int(num2)
    if expr=='*': return int(num1)*int(num2)
    if expr=='/': return int(int(num1)/int(num2))

def calculate(s: str) -> int:
    oper = '+-*/'
    s = s.replace(' ','')
    sss = list(s)
    i = 1
    j,k = 1,1
    ss = [sss[0]]
    for v in range(1,len(sss)):
        if sss[v] in oper or sss[v-1] in oper:
            ss.append(sss[v])
        else:
            ss[-1]+=sss[v]
    print(ss)
    while '/' in ss or '*' in ss:
        if ss[i]=='/' or ss[i]=='*':
            re = calc(ss[i-1], ss[i], ss[i+1])
            second = ss[i+2:]
            first = ss[:i-1]
            first.append(str(re))
            ss = first+second
            i-=3
        i+=1
            
   
    while '+' in ss or '-' in ss:
        if ss[k]=='-' or ss[k]=='+':
            re = calc(ss[k-1], ss[k], ss[k+1])
            second = ss[k+2:]
            first = ss[:k-1]
            first.append(str(re))
            ss = first+second
            k-=3
        k+=1
    
    return int(ss[-1])

# a = " 3+5 / 2 "
# a = "123/123+21234/23-1234*234*23/23*1234-234123+234"
a = " 1+25/3+6/42"
a = "1+2*5/3+6/4*2"












from math import ceil

def calc(num1, expr, num2):
    if expr=='+':return int(num1)+int(num2)
    if expr=='-':return int(num1)-int(num2)
    if expr=='*': return int(num1)*int(num2)
    if expr=='/': return int(int(num1)/int(num2))
def calculate(s: str) -> int:
    oper = '+-*/'
    s = s.replace(' ','')
    sss = list(s)
    i = 0
    ss = [sss[0]]
    numbers = []
    for v in range(1,len(sss)):
        if sss[v] in oper or sss[v-1] in oper:ss.append(sss[v])
        else:ss[-1]+=sss[v]
    while len(ss)>1:
        if ss[i] in oper:
            if ss[i]=='*' or ss[i]=='/':
                if len(numbers):
                    n1 = ss[i-1]
                    temp = int(n1)-numbers[-2]
                    re = calc(temp,ss[i],ss[i+1])
                    re-=ss[i+1]
                    second = ss[i+2:]
                    first = ss[:i-1]
                    first.append(str(re))
                    ss = first+second
                    numbers.append(re)
                    i-=2
                else:
                    re = calc(ss[i-1], ss[i], ss[i+1])
                    second = ss[i+2:]
                    first = ss[:i-1]
                    first.append(str(re))
                    ss = first+second
                    
                    # numbers.append(re)
                    i-=2
            else:
                nn = ss[i-1]
                re = calc(nn, ss[i], ss[i+1])
                second = ss[i+2:]
                first = ss[:i-1]
                first.append(str(re))
                ss = first+second
                if not len(numbers):
                        numbers.append(int(nn))
                i-=2
                numbers.append(re)

        i+=1
    return int(ss[-1])


a = "3+3-23+32-3+2*2+12-4+14*2"
a="2*3*4"
a = "1+2*5/3+6/4*2"


num = [1,3,11,5,11,6,]


# print(int(-5/2)+5)

# print(calculate(a))







def calculeat(s:str):

    i = 0

    cur=prev=res = 0

    cur_oper = '+'

    
    while i < len(s):
        cur_char = s[i]
        if cur_char.isdigit():
            while i<len(s) and s[i].isdigit():
                cur = cur * 10 + int(s[i])

                i += 1

            i-=1

            if cur_oper == '+':
                res+=cur
                prev=cur
            elif cur_oper=='-':
                res-=cur
                prev=cur
            elif cur_oper=='*':
                res-=prev
                res+=prev*cur

                prev = cur*prev
            else:
                res-=prev
                res+=int(prev/cur)
                prev = int(prev/cur)
            cur = 0
        elif cur_char!=' ':
            cur_oper = cur_char

        i+=1
    return res












import time

def timesss(ineer_func):
    def wrapper_func(*args, **kwargs):
        start  = time.time()
        ineer_func(*args, **kwargs)
        end = time.time()
        duration = end - start
        print(f"Executed {ineer_func.__name__} in {duration:.3f} secs")
    return wrapper_func

@timesss
def do_somethink(n: int):
    for x in range(n):
        for u in range(n):
            print(x*u)

# tooo_a = timesss(do_somethink)
# tooo_a(12)
# do_somethink(12)



# def calc(arr: list[int]):
#     oper = "+-/*"
#     i = 0
#     while len(arr)>1:
#         if arr[i] in oper:













def numRescueBoats(people: list[int], limit: int) -> int:
    res = 0
    people.sort()

    while len(people)>0:
        if people[-1]==limit:
            res+=1
            people.pop(len(people)-1)
        
                
    return res


def minimizeArrayValue(nums: list[int]) -> int:

    while max(nums)!=nums[0]:

        a = nums.index(max(nums))

        nums[a]-=1
        nums[a-1]+=1


    return nums[0]





