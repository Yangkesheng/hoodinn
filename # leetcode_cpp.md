# 

## 31 
* **思路** 
1. 从后往前遍历，找到第一递减（从后往前看递减的）的数字i（只有用i之后比这个数字大的交换，才能获得比当前大的数字），没有找到跳到第四部
2. 再从后往前遍历，找到第一比i大的值j，这两个值交换（由于第一步的操作，i+1到end必定是降序的）
3. 对i+1之后到end的值，进行排序，保证这段区间的数是最下的数是最下的
4. 到这表明整个数列是递减的，没有一个更大的了，排序整个数组就可以了

```cpp
class Solution {
public:
    void nextPermutation(vector<int>& nums) {
        if (nums.size() < 2) return;

        int pos = -1;
        for (int i = nums.size()-2; i >= 0; i--) {
            if (nums[i] < nums[i+1]) {
                pos = i;
                break;
            }
        }

        if (pos != -1) {
            for (int i = nums.size()-1; i >= 0; i--) {
                if (nums[i] > nums[pos]) {
                    swap(nums[i], nums[pos]);

                    // sort(&nums[0]+pos+1, &nums[0] + nums.size());
                    sort(&nums[pos+1], &nums[nums.size()]);
                    break;
                }
            }
        } else {
            sort(nums.begin(), nums.end());
        }
    }
};
```

## 32
* **思路** 动态规划
1. 遍历字符串，如果当前的‘)’前以为的如果是‘(’,则 **dp[i+1]** = dp[i-1] + 2
2. eg:"(())" 当i=3时，检查s[i- dp[i]-1] (i要减去一个数，注意越界情况，所以要校验 i-dp[i]-1 > 0) 是否能匹配，若匹配上了则字符长度连接起来了，dp[i+1] = dp[i] + 2 + dp[i-dp[i]-1]
```cpp
class Solution {
public:
    int longestValidParentheses(string s) {
        if (s.size() < 2)
            return 0;

        vector<int> dp(s.size() + 1, 0);
        for (int i = 1; i < s.size(); i++) {
            if (s[i] == ')') {
                if (s[i-1] == '(') {
                    dp[i + 1] = dp[i - 1] + 2;
                } else {
                    if (i-dp[i]-1 >= 0 && s[i-dp[i]-1] == '(') //注意这个=0,条件
                        dp[i + 1] = dp[i] + dp[i - dp[i] - 1] + 2;
                }
            }
        }

        int max = dp[0];
        for (int i = 1; i < dp.size(); i++) {
            max = std::max(max, dp[i]);
        }

        return max;
    }
}; 
```

## 33 
* **思路** 折半查找
1. 判断中间值是不是等于要找的值，不是继续
2. 根据中间值判断出递增的区间，如果不再递增区间内，折半循环去找
```cpp
class Solution {
public:
    int search(vector<int>& nums, int target) {
        int left = 0, right = nums.size() - 1;

        while (left <= right) {//等于号，注意
            int mid = (left + right) >> 2;

            if (target == nums[mid]){
                return mid;
            } else if (nums[mid] < nums[right]) { //右边递增
                if (nums[mid] < target && target <= nums[right]) {
                    left = mid+1;
                } else {
                    right = mid-1;
                }
            } else {//左边递增
                if (nums[left] <= target && target <= nums[mid])
                    right = mid-1;
                else
                    left = mid+1;
            }
        }

        return -1;
    }
}
```

## 34
* **思路** 题目要求时间复杂度为 O(log n)，这显然是要用到折办查找
1. 先用一遍折办查找，找最左边的
2. 再用一遍折半查找，找最右边的

```cpp
class Solution {
public:
    vector<int> searchRange(vector<int>& nums, int target) {
        vector<int> res(2, -1);

        int left = 0, right = nums.size() - 1;
        while (left <= right) {            //等号
            int mid = (left + right) >> 1;

            if (target == nums[mid]) {    
                res[0] = mid;              //记录
                right = mid - 1;           //继续找左边
            }
            else if (target < nums[mid]) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }

        left = 0;
        right = nums.size()-1;
        while (left <= right) {
            int mid = (left + right) >> 1;

            if (target == nums[mid]) {
                res[1] = mid;              //记录
                left = mid + 1;            //继续找右边
            } else if (target < nums[mid]) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }

        return res;
    }
};
```
## 39
* **思路** 深度优先遍历
1. 便利当前列表，减去当前值，新值重复
```cpp
class Solution {
public:
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        dfs(candidates, 0, target, vector<int>());

        return res;
    }

private:
    void dfs(vector<int> & candidates, int pos, int target, vector<int> now) {
        if (target < 0) return;

        //一个解
        if (target == 0) {
            vector<int> temp(now);
            res.push_back(temp);

            return;
        }

        for (int i = pos; i < candidates.size(); i++){
            if (candidates[i] > target)
                continue;

            now.push_back(candidates[i]);
            dp(candidates, i, target - candidates[i], now);
            now.pop_back();
         }
    }

private:
    vector<vector<int>> res;
};
```

##42
* **思路** 每一个柱子的相邻的矮的决定了乘水的多少。找到了池的较矮的一边，那么这个池的呈水量就决定了。（思想就是去确定池子矮的一边，这个问题就解决了）
1. 找到两边较矮的开始计算
2. 当前位置的的呈水量等于较矮的一边的最高减去当前高度

```cpp
#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

class Solution {
public:
    int trap(vector<int>& height) {
        int sum = 0, maxLeft = 0, maxRight = 0, left = 1, right = height.size() - 2;

        while (left <= right) {
        // for (int i = 0; i < height.size()-1; i++) {
            if (height[left-1] < height[right+1]) {
                maxLeft = max(maxLeft, height[left - 1]);

                if (maxLeft > height[left])
                    sum += maxLeft - height[left];

                left++;
            } else {
                maxRight = max(maxRight, height[right + 1]);
                if (maxRight > height[right]) 
                    sum += maxRight - height[right];

                right--;
            }
        }

        return sum;
    }
};
```

## 46 
* **思路**dfs
1. for循环 dp，记录或查找已经插入的数据就可以

```cpp
class Solution {
public:
    vector<vector<int>> permute(vector<int>& nums) {
        vector<int> res;
        dp(nums, res);

        return ans;
    }

private:
    void dp(vector<int>& nums, vector<int>& res) {
        for (int i = 0; i < nums.size(); i++) {
            if (res.size() == nums.size()) { //先判断
                vector<int> t(res);
                ans.push_back(t);

                return;
            }

            if (find(res.begin(), res.end(), nums[i]) != res.end()) continue;
            
            res.push_back(nums[i]);
            dp(nums, res);
            res.pop_back();
        }
    }

private:
    vector<vector<int>> ans;
};
```
## 48
1. 沿对角线交换
2. 上下交换
```cpp
class Solution {
public:
    void rotate(vector<vector<int>>& matrix) {
        int n = matrix.size();

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i+j < n) {
                    swap(matrix[i][j], matrix[n-j-1][n-i-1]);
                }
            }
        }

        for (int i = 0; i < n / 2; i++) {
            for (int j = 0; j < n; j++) {
                swap(matrix[i][j], matrix[n-i-1][j]);
            }
        }
    }
};
```

## 53 
1. 比较当前值和当前值加上前面的总和（记录这个一段的和， eg：2, -1, 2）
2. 比较当前总和最大值，记录最大值

```cpp
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int m = nums[0], now = nums[0];

        for (int i = 1; i < nums.size(); i++) {
            now = max(nums[i], now + nums[i]);

            m = max(now, m);
        }

        return m;
    }
};

```

## 55 
* **思路** 贪心算法
1. 当前的位置已经大于了可以到达位置，不可达到
2. 记录当前可达的最远点 

```cpp
class Solution {
public:
    bool canJump(vector<int>& nums) {
        int can_arrived = 0;

        for (int i = 0; i < nums.size(); i++) {
            if (i > can_arrived)
                return false;

            can_arrived = max(i + nums[i], can_arrived);
        }

        return true;
    }
};
```
## 56
* **思路** 
1. 把区间的头和尾分开
2. i的区间尾在i+1区间之间，那么继续判断下一个 
```cpp
class Solution {
public:
    vector<vector<int>> merge(vector<vector<int>>& intervals) {
        vector<int> start, end;
        vector<vector<int>> ans;

        for (auto it = intervals.begin(); it != intervals.end(); it++) {
            start.push_back((*it)[0]);
            end.push_back((*it)[1]);
        }

        sort(start.begin(), start.end());
        sort(end.begin(), end.end());

        for (int i = 0; i < start.size(); i++) {
            vector<int> res;
            res.push_back(start[i]);

            while (i+1 < start.size() && start[i+1] <=  end[i] && end[i] <= end[i+1]) {
                i++;
            }

            res.push_back(end[i]);
            ans.push_back(res);
        }

        return ans;
    }
};
```
## 62 
* **思路** 动态规划 dp[i][j] = dp[i - 1][j] + dp[i][j - 1]  (每个点线路等于顶上和左侧的和)
1. 预处理边界，i=1的情况
2. 根据动态方程填写，注意特殊情况（j=1）
```cpp
class Solution {
public:
    int uniquePaths(int m, int n) {
        vector<vector<int>> dp;

        for (int i = 0; i < m; i++) {
            if (i != 0)
                dp.push_back(vector<int>(n, 0));
            else
                dp.push_back(vector<int>(n, 1));
        }

        for (int i = 1; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (j != 0) {
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
                }
                else {
                    dp[i][j] = 1;
                }
            }
        }

        return dp[m - 1][n - 1];
    }
};
```

## 70
* **思路** 找规律，斐波那契数列
```cpp
class Solution {
public:
    int climbStairs(int n) {
        if (n < 2)
            return n;

        int n1 = 1, n2 = 2, ans;
        for (int i = 3; i <= n; i++) {
            ans = n1 + n2;

            n1 = n2;
            n2 = ans;
        }

        return ans;
    }
};
```

## 72 
* **思路** 动态规划
1. dp[i+1][j+1] = min(dp[i+1][j], dp[i-1][j], dp[i+1][j-1]) (就是它相邻的3个点的最小值)

```cpp
class Solution {
public:
    int minDistance(string word1, string word2) {
        //处理dp初始化，处理极限情况，即wordi、word2之一为空
        for (int i = 0; i < word1.size()+1; i++) {
            vector<int> row(word2.size() + 1, 0);

            for (int j = 0; j < word2.size()+1; j++) {
                if (i == 0)
                    row[j] = j;
                else if (j == 0)
                    row[j] = i;
            }

            dp.push_back(row);
        }


        for (int i = 0; i < word1.size(); i++) {
            for (int j = 0; j < word2.size(); j++) {
                if (word1[i] != word2[j]) {
                    //插入：虚拟操作，只是相当于消耗了一个word2字符）所以当前的操作次数就等于word2[j-1]的操作次数（就是相当于把j删除掉）
                    dp[i + 1][j + 1] = dp[i + 1][j]; 

				    //删除：（虚拟操作，只是相当于消耗了一个word1字符）word1[i]当前位置删除，相当于查看前面的位置操作次数
                    dp[i + 1][j + 1] = min(dp[i + 1][j + 1], dp[i][j+1]);

				    //置换：操作次数等于，(word1[i-1] != word2[j-1]情况
                    dp[i + 1][j + 1] = min(dp[i + 1][j + 1], dp[i][j]);

                    //上面3种情况去最小的情况，加上本次操作，就是当前匹配成功的操作次数
                    dp[i + 1][j + 1] += 1;
                } else {
                    dp[i + 1][j + 1] = dp[i][j];
                }
            }
        }

        return dp[word1.size()][word2.size()];
    }

private:
    vector<vector<int>> dp;
};
```

## 76
* **思路** 滑动窗口
1. 记录t中字符串，带个数的，t中有可能重复
2. 遍历s，出现t中的字符‘a’，a的个数-1，当a数量为大于0表示一次成功，n(记录匹配长度)+1，否则是前面的字段已经匹配过了n就不加了，但是i的记录还是去减少

```cpp
#include <iostream>
#include <map>
#include <string>

using namespace std;

class Solution {
public:
    string minWindow(string s, string t) {
        map<char, int> cache;

        for (int i = 0; i < t.size(); i++) 
            cache[t[i]]++;

        int l = 0 , n = 0;
        string res;

        for (int r = 0; r < s.size(); r++) { 
            auto it = cache.find(s[r]);

            if (it != cache.end()) {
                if (it->second > 0) {
                    n++;
                }

                cache[s[r]]--;
            }

            //l=r 边界情况，t只有一个字符
            while (n == t.size() && l <= r) {
                if (res == "" || res.size() > r-l+1) {
                    res = s.substr(l, r - l + 1);
                }

                auto it2 = cache.find(s[l]);
                if (it2 != cache.end()) {
                    it2->second += 1;

                    if (it2->second > 0)
                        n--;
                }

                l++;
            }
        }

        return res;
    }
};
```

## 78
1. 先插入一个空
2. 每一个数字插入之前先copy解追加之后，再插入
 
```cpp
class Solution {
public:
    vector<vector<int>> subsets(vector<int>& nums) {
        ans.push_back(vector<int>());

        for (int i = 0; i < nums.size(); i++) {
            dp(nums[i]);
        }

        return ans;
    }

private:
    void dp(int num) {
        for (int i = 0, m = ans.size(); i < m; i++) {
            vector<int> res(ans[i]);

            res.push_back(num);
            ans.push_back(res);
        }
    }
    
private:
    vector<vector<int>> ans;
};
```

## 79
* **思路** dfs 
1. 设置一个缓存记录走过的路
2. 当首字母匹配成功开始，递归的去调用 

```cpp
class Solution {
public:
    bool exist(vector<vector<char>>& board, string word) {
        for (int i = 0; i < board.size(); i++) 
            seen.push_back(vector<bool> (board[i].size(), false));

        for (int i = 0; i < board.size(); i++) {
            for (int j = 0; j < board[i].size(); j++) {
                if (board[i][j] == word[0] && dp(i, j, board, word.substr(1))) 
                    return true;
            }
        }

        return false;
    }

private:
    bool dp(int i, int j, vector<vector<char>>& board, string word) {
        if (!word.size())
            return true;

        seen[i][j] = true;

        for (int n = 0; n < ii.size(); n++) {
            if (0 <= i+ii[n] && i+ii[n] < board.size() 
            && 0 <= j+jj[n] && j+jj[n] < board[i].size()
            && !seen[i+ii[n]][j+jj[n]] 
            && board[i+ii[n]][j+jj[n]] == word[0]) {
                if (dp(i + ii[n], j + jj[n], board, word.substr(1)))
                    return true;
            }
        }

        seen[i][j] = false;

        return false;
    }

private:
    vector<vector<bool>> seen;
    vector<int> ii = {0, 1, 0, -1}, jj = {-1, 0, 1, 0};
};
```


## 84
* **思路** 单调递增栈
1. 给两边增加两个0,为了解决边界的高度计算
2. 遍历高度，碰到变矮的数字，没法连续了，开始计算它左侧比它高的到地方（eg: 0,1,2,0。这里处理1, 2这里，先算2这个位置，再算1,2这个组合）

```cpp
class Solution {
public:
    int largestRectangleArea(vector<int>& heights) {
        stack<int> pos;
        heights.insert(heights.begin(), 0);
        heights.insert(heights.end(), 0);

        int res = 0;
        for (int i = 0; i < heights.size(); i++) {
            while (!pos.empty() && heights[pos.top()] > heights[i]) { //保证单调递增的栈
                int h =  heights[pos.top()];
                //保证单调递增的栈
                pos.pop();

                //i-1（表示从i的左侧）到pos.top()的距离
                int w = i-1 - pos.top();
                res = max(res, w * h);
            }

            pos.push(i);
        }

        return res;
    }
};
```
## 85 
* **思路** 动态规划
1. 设置dp[]一维数组，记录当前竖列连续的1
2. 遍历数组时，当是1时，可以根据dp数组找到h（高度），然后开始倒序遍历，找可能的宽度
3. 如果不是1,断掉当前的dp中连续的高度。

```cpp
class Solution {
public:
    int maximalRectangle(vector<vector<char>>& matrix) {
        if (!matrix.size())
            return 0;

        vector<int> dp(matrix[0].size(), 0);
        int res = 0;

        for (int i = 0; i < matrix.size(); i++) {
            for (int j = 0; j < matrix[i].size(); j++) {

                if (matrix[i][j] == '1') {
                    dp[j] += 1;

                    int h = dp[j];
                    for (int k = j; k >= 0 && matrix[i][k] == '1'; k--) {
                        h = min(h, dp[k]);

                        res = max(res, (j - k + 1) * h);
                    }
                } else {
                    dp[j] = 0;
                }
            }
        }

        return res;
    }
};
```

## 94 
* **思路** 递归（若用非递归，就需要用到栈）

```cpp
class Solution {
public:
    vector<int> inorderTraversal(TreeNode* root) {
        dp(root);

        return res;
    }

private:
    void dp(TreeNode* root) {
        if (root == NULL)
            return;

        inorderTraversal(root->left);
        res.push_back(root->val);
        inorderTraversal(root->right);
    }

    vector<int> res;
};
```
## 96
* **思路**
假设n个节点存在二叉排序树的个数是G(n)，令f(i)为以i为根的二叉搜索树的个数即有:G(n) = f(1) + f(2) + f(3) + f(4) + ... + f(n)。n为根节点，当i为根节点时，其左子树节点个数为[1,2,3,...,i-1]，右子树节点个数为[i+1,i+2,...n]，所以当i为根节点时，其左子树节点个数为i-1个，右子树节点为n-i，即f(i) = G(i-1)*G(n-i),上面两式可得:G(n) = G(0)*G(n-1)+G(1)*(n-2)+...+G(n-1)*G(0)

```cpp
class Solution {
public:
    int numTrees(int n) {
        vector<int> dp(n + 1, 0);
        dp[0] = 1;

        for (int i = 1; i <= n; i++) {
            for (int j = 0; j < i; j++) {
                dp[i] += dp[j] * dp[i - j - 1];
            }
        }

        return dp[n];
    }
};
```
