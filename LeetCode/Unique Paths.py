"""
uniquePath1
A robot is located at the top-left corner of a m x n grid (marked 'Start' in the diagram below).

The robot can only move either down or right at any point in time. The robot is trying to reach the bottom-right corner of the grid (marked 'Finish' in the diagram below).

How many possible unique paths are there?
"""


class Solution(object):
    def uniquePathsA(self, m, n):
        """

        :param m:
        :param n:
        :return:
        """
        dp = [[0]*n for x in range(m)]
        dp[0][0] = 1
        for x in range(m):
            for y in range(n):
                if x + 1 <m:
                    dp[x+1][y] += dp[x][y]
                if y + 1 < n:
                    dp[x][y+1] += dp[x][y]
        return dp[m-1][n-1]

    def uniquePathsB(self, m, n):
        """

        :param m:
        :param n:
        :return:
        """
        if m < n:
            m, n = n, m
        dp = [n] * n
        dp [0] = 1
        for x in range(m):
            for y in range(n - 1):
                dp[y + 1] += dp[y]
        return dp[n - 1]

    #m行n列, 需要往下走m-1步, 往右走n-1步, 也就是求C(m-1+n-1, n-1)或C(m-1+n-1, m-1)。

    # @return an integer
    def uniquePathsC(self, m, n):
        N = m - 1 + n - 1
        K = min(m, n) - 1
        # calculate C(N, K)
        res = 1
        for i in xrange(K):
            res = res * (N - i) / (i + 1)
        return res


"""
Unique Paths II
Now consider if some obstacles are added to the grids. How many unique paths would there be?

An obstacle and empty space is marked as 1 and 0 respectively in the grid.

"""





"""
Minimum Path Sum
Given a m x n grid filled with non-negative numbers, find a path from top left to bottom right which minimizes the sum of all numbers along its path.

Note: You can only move either down or right at any point in time.

"""


"""
Dungeon Game
The demons had captured the princess (P) and imprisoned her in the bottom-right corner of a dungeon. The dungeon consists of M x N rooms laid out in a 2D grid. Our valiant knight (K) was initially positioned in the top-left room and must fight his way through the dungeon to rescue the princess.

The knight has an initial health point represented by a positive integer. If at any point his health point drops to 0 or below, he dies immediately.

Some of the rooms are guarded by demons, so the knight loses health (negative integers) upon entering these rooms; other rooms are either empty (0's) or contain magic orbs that increase the knight's health (positive integers).

In order to reach the princess as quickly as possible, the knight decides to move only rightward or downward in each step.


Write a function to determine the knight's minimum initial health so that he is able to rescue the princess.

For example, given the dungeon below, the initial health of the knight must be at least 7 if he follows the optimal path RIGHT-> RIGHT -> DOWN -> DOWN.
"""


class Solution(object):
    def __init__(self):
        self.depth = 0
        self.nestedList = []
    def depthSum(self, nestedList):
        """
        :type nestedList: List[NestedInteger]
        :rtype: int
        """
        nest = nestedList

        if isinstance(type(nest), int):
            return self.depth + 1
        else:
            for n in nest:
                if isinstance(type(n), int):
                    self.depth = self.depth + 1
                else:
                    self.depth = self.depth + self.depthSum(n)
        return self.depth


class Solution(object):
    def depthSum(self, nestedList):
        """
        :type nestedList: List[NestedInteger]
        :rtype: int
        """
        if len(nestedList) == 0: return 0
        stack = []
        sum = 0
        for n in nestedList:
            stack.append((n, 1))
        while stack:
            next, d = stack.pop(0)
            if next.isInteger():
               sum += d * next.getInteger()
            else:
                for i in next.getList():
                    stack.append((i,d+1))
        return sum