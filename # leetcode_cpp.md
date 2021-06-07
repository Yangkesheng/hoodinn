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
                if (i+j <= n/2) {
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