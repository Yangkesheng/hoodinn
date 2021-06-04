#include <iostream>
#include <vector>
#include <map>
#include <algorithm>

using namespace std;

class Solution {
public:
    vector<vector<string>> groupAnagrams(vector<string>& strs) {
        for (int i = 0; i < strs.size(); i++) {
            string t(strs[i]);
            sort(t.begin(), t.end());

            ans[t].push_back(strs[i]);
        }

        vector<vector<string>> rtn;
        for (auto it = ans.begin(); it != ans.end(); it++) {
            rtn.push_back(it->second);
        }

        return rtn;
    }

private:
    map<string, vector<string>> ans;
};

/* #include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

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
            if (res.size() == nums.size()) {
                vector<int> t(res);
                ans.push_back(t);

                return;
            }
            // if (seen[i] == 1) continue;
            if (find(res.begin(), res.end(), nums[i]) != res.end()) continue;
            
            res.push_back(nums[i]);
            dp(nums, res);
            res.pop_back();
        }
    }

private:
    vector<vector<int>> ans;
};

int main(int argc, const char** argv) {
    Solution test;
    vector<int> test1 = {1, 2, 3};
    test.permute(test1);

    return 0;
} */
/* 
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

 */
/* #include <iostream>
#include <vector> 
#include <cstring>

using namespace std;

class Solution {
public:
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        dp(candidates, 0, target, vector<int>());

        return res;
    }

private:
    void dp(vector<int> & candidates, int pos, int target, vector<int> now) {
        if (target < 0) return;

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

struct {
    short a;
    short b;
    short c;
} A;

struct {
    long a;
    short b;
} B;

int main() {
    Solution test;
    int ca[]  = {1, 2, 3, 4};
    int *p = ca;
    // test.combinationSum(ca, 5);

    // cout << "sizeof(short)" << sizeof(short) << endl;
    // cout << sizeof(A) << endl;

    // cout << "sizeof(long)" << sizeof(long) << endl;
    // cout << "sizeof(bool)" << sizeof(bool) << endl;
    // cout << sizeof(B) << endl;

    char *ss = "asd";

    char str[] = "asdfafa";
    cout << sizeof(ss) << endl;
    cout << sizeof(p) << endl;
    cout << sizeof(ca) << endl;
    cout << sizeof(*ss) << endl;
    cout << *ss << endl;
    cout << *ca << endl;

    cout << strlen(ss) << endl;
    cout << strlen(str) << endl;

    return 0;
}
 */
/* #include <iostream>
#include <vector>

using namespace std;

class Solution {
public:
    vector<int> searchRange(vector<int>& nums, int target) {
        vector<int> res(2, -1);

        int left = 0, right = nums.size() - 1;
        while (left <= right) {
            int mid = (left + right) >> 1;

            if (target == nums[mid]) {
                res[0] = mid;
                right = mid - 1;
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
                res[1] = mid;
                left = mid + 1;
            } else if (target < nums[mid]) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }

        return res;
    }
};

int main() {
    Solution test;
    vector<int> t(3, 1);
    auto res = test.searchRange(t, 1);

    cout << res[0] << ", " << res[1] << endl;

    return 0;
}
 */
/* #include <iostream>
#include <vector>

using namespace std;

class Solution {
public:
    int search(vector<int>& nums, int target) {
        int left = 0, right = nums.size() - 1;

        while (left < right) {
            int mid = (left + right) >> 2;

            if (target == nums[mid]){
                return mid;
            } else if (nums[mid] < nums[right]) {
                if (nums[mid] < target && target <= nums[right]) {
                    left = mid+1;
                } else {
                    right = mid-1;
                }
            } else {
                if (nums[left] <= target && target <= nums[mid])
                    right = mid-1;
                else
                    left = mid+1;
            }
        }

        return -1;
    }
}
 */
/* #include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

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
                    if (i-dp[i]-1 >= 0 && s[i-dp[i]-1] == '(')
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

int main() {
    Solution test;

    cout << test.longestValidParentheses("(()())") << endl;
}
 */
/* 
// 输入：nums = [1,2,3]
// 输出：[1,3,2]

// [4,2,0,2,3,2,0]
// [4,2,0,2,2,0,3]
// [4,2,0,2,0,2,3]
// [4,2,0,3,0,2,2]

#include <iostream>
#include <vector>
#include <algorithm>

void myPrintf(vector<int>& nums) {
    for (vector<int>::const_iterator it = nums.begin(); it != nums.end(); it++) {
        if (it != nums.begin()) 
            cout << ", ";

        cout << *it;
    }

    cout << endl;
}

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


int main() {
    // vector<int> test = {1,2,3};
    Solution test1;
    vector<int> test = {4,2,0,2,3,2,0};

    myPrintf(test);
    test1.nextPermutation(test);
    myPrintf(test);

    vector<int> test2 = {3,2,1};

    myPrintf(test2);
    test1.nextPermutation(test2);
    myPrintf(test2);

    return 0;
}
 */
/* 
#include <iostream>
#include <vector>
#include <limits.h>

using namespace std;

struct ListNode {
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
} * ptr;

class Solution {
public:
    ListNode* mergeKLists(vector<ListNode*>& lists) {
        int j = -1, min = INT_MAX;

        for (int i = 0; i < lists.size(); i++) {
            if (lists[i] && lists[i]->val < min) {
                min = lists[i]->val;
                j = i;
            }
        }

        if (j == -1) {
            return NULL;
        } 

        ListNode* newNode = lists[j];
        lists[j] = lists[j]->next;

        newNode->next = mergeKLists(lists);

        return newNode;
    }
};

int main() {
    ListNode t2(2);
    ListNode t1(1, &t2);
    ListNode t3(3);
    //ListNode t4(4);

    ListNode* head1 = &t1;
    ListNode* head2 = &t3;

    vector<ListNode*> lists{head1, head2};

    Solution test;
    ListNode* res = test.mergeKLists(lists);

    return 0;
}
 */

/* 
#include <iostream>
#include <vector>

using namespace std;

class Solution {
public:
    vector<string> generateParenthesis(int n) {
        dfs(n, n, "");

        return res;
    }

private:
    vector<string> res;
    
    void dfs(int left, int right, string curStr) {
        if (left > right) return; //减少无意义的递归

        if (left == 0 && right == 0) {
            cout << curStr << endl;
            res.push_back(curStr);
        }
        
        if (left > 0)
            dfs(left-1, right, curStr+"(");
        
        if (right > 0)
            dfs(left, right-1, curStr+")");
    }
};

int main() {
    Solution test;
    test.generateParenthesis(3);

    return 0;
}
 */
/* 
#include <iostream>

using namespace std;

struct ListNode {
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};

class Solution {
public:
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
        if ((!l1 && !l2) || (l1 && !l2)) return l1;
            
        if (!l1 && l2) return l2;

        ListNode* newNode;
        if (l1->val <= l2->val) {
            l1->next = mergeTwoLists(l1->next, l2);
            return l1;
        } else {
            l2->next = mergeTwoLists(l1, l2->next);
            return l2;
        }
    }
};
 */
/* 
#include <iostream>

using namespace std;

struct ListNode {
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};

class Solution {
public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        if (!head) return NULL;

        head->next = removeNthFromEnd(head->next, n);

        count++;
        if (count == n) {
            head = head->next;
        }

        return head;
    }

private:
    int count = 0;
};
 */
/* 
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        vector<vector<int> > res;

        if (nums.size() < 3) return res;

        sort(nums.begin(), nums.end());

        for (int i = 0; i < nums.size(); i++) {
            if (i && nums[i] == nums[i-1]) continue;

            for (int j = i+1, end = nums.size()-1; j < end; j++) {
                if (j > i+1 && nums[j] == nums[j-1]) continue;

                while(j < end-1 && nums[i] + nums[j] + nums[end] > 0)
                    end--;

                if (nums[i] + nums[j] + nums[end] == 0)
                    res.push_back({nums[i], nums[j], nums[end]});
            }
        }
       
       return res;
    }
};

void myPrintf(vector<vector<int>> nums) {
    cout << "["; 
    for (int i = 0; i < nums.size(); i++) {
        if (i != 0) cout << ", ";

        cout << "[";
        for (int j = 0; j < nums[i].size(); j++) {
            if (j != 0) cout << ", ";

            cout << nums[i][j];
        }
        cout << "]";
    }
    cout << "]" << endl; 
}

int main() {
    Solution test;

    vector<int> test1 {0,};
    vector<int> test2 {0,0,0};
    vector<int> test3 {0,0,0,0};
    vector<int> test4 {-1, 0, 1, 2, -1, -4};
    vector<int> test5 {-12, 4, 12, -4, 3, 2, -3, 14, -14, 3, -12, -7, 2, 14, -11, 3, -6, 6, 4, -2, -7, 8, 8, 10, 1, 3, 10, -9, 8, 5, 11, 3, -6, 0, 6, 12, -13, -11, 12, 10, -1, -15, -12, -14, 6, -15, -3, -14, 6, 8, -9, 6, 1, 7, 1, 10, -5, -4, -14, -12, -14, 4, -2, -5, -11, -10, -7, 14, -6, 12, 1, 8, 4, 5, 1, -13, -3, 5, 10, 10, -1, -13, 1, -15, 9, -13, 2, 11, -2, 3, 6, -9, 14, -11, 1, 11, -6, 1, 10, 3, -10, -4, -12, 9, 8, -3, 12, 12, -13, 7, 7, 1, 1, -7, -6, -13, -13, 11, 13, -8};
    myPrintf(test.threeSum(test1));
    myPrintf(test.threeSum(test2));
    myPrintf(test.threeSum(test3));
    myPrintf(test.threeSum(test4));
    myPrintf(test.threeSum(test5));

    return 0;
}
 */
/* 
#include <iostream>
#include <map>

using namespace std;

class Solution {
public:
    string intToRoman(int num) {
        map<int, string> weight = {{1000, "M"}, {900, "CM"}, {500, "D"}, {400, "CD"}, {100, "C"}, {90, "XC"}, {50, "L"}, {40, "XL"}, {10, "X"}, {9, "IX"}, {5, "V"}, {4, "IV"}, {1, "I"},};

        string rtn = "";
        for (map<int,string>::reverse_iterator it = weight.rbegin(); it != weight.rend(); it ++ ) {
            for (int i = num / it->first; i != 0; i--)
                rtn += it->second;
            
            num %= it->first;
        }

        return rtn;
    }
};

int main() {
    Solution test;

    cout << test.intToRoman(3) << endl;
    cout << test.intToRoman(4) << endl;
    cout << test.intToRoman(1994) << endl;
    cout << test.intToRoman(3999) << endl;

    return 0;
}
 */
/* 
#include <iostream>
#include <vector>

using namespace std;

class Solution {
public:
    int maxArea(vector<int>& height) {
        int max = 0;

        for (int i = 0, j = height.size()-1; i < j;) {
            if (height[i] <= height[j]) {
                max = max <= height[i] * (j-i) ? height[i]*(j-i) : max;

                i++;
            } else {
                max = max <= height[j] * (j-i) ? height[j]*(j-i) : max;

                j--;
            }
        }

        return max;
    }
};

int main() {
    Solution test;
fmt.Println(maxArea([]int{1, 8, 6, 2, 5, 4, 8, 3, 7}))
	fmt.Println(maxArea([]int{1, 1}))
	fmt.Println(maxArea([]int{4, 3, 2, 1, 4}))
	fmt.Println(maxArea([]int{1, 2, 1}))

    vector<int> test1{1, 8, 6, 2, 5, 4, 8, 3, 7};
    vector<int> test2{1, 2,1};
    vector<int> test3{1, 2,1};
    vector<int> test4{1, 2,1};
    cout << test.maxArea(test1) << endl;

    return 0;
}
 */
/* #include <iostream>
#include <vector>


using namespace std;

class Solution {
public:
    bool isMatch(string s, string p) {
        if (p.empty()) return s.empty();

        vector<vector<int>> dp(s.size()+1, vector<int>(p.size()+1, 0));

        dp[0][0] = 1;
        for (int j = 1; j <= p.size(); j++) {
            if (p[j-1] != '*') 
                continue;
            else 
                dp[0][j] = dp[0][j-2];
        }

        for (int i = 1; i <= s.size(); i++) {
            for (int j = 1; j <= p.size(); j++) {
                if (p[j-1] == s[i-1] || p[j-1] == '.') {
                    dp[i][j] = dp[i-1][j-1];
                } else if (p[j-1] == '*') {
                    dp[i][j] = dp[i][j-2];

                    if (p[j-2] == '.' || p[j-2] == s[i-1])
                        dp[i][j] = dp[i][j] | dp[i-1][j];
                }
            }   
        }

        return dp[s.size()][p.size()];
    }
};

int main() {
    Solution test;

    cout << test.isMatch("ab", ".*") << endl;
	// fmt.Println(isMatch("aab", "c*a*b"))
	// fmt.Println(isMatch("aa", "a*"))
	// fmt.Println(isMatch("ab", ".*c"))
	// fmt.Println(isMatch("aa", "a"))
	// fmt.Println(isMatch("mississippi", "mis*is*p*."))
	// fmt.Println(isMatch("aaa", "ab*a"))
	cout << test.isMatch("aab", "ab*a*c*") << endl;
	cout << test.isMatch("aaba", "ab*a*c*a") << endl;
    cout << test.isMatch("aaa", "ab*a*c*a") << endl;

    return 0;
}
 */
/* 
#include <iostream>
#include <cmath>

using namespace std;

class Solution {
public:
    bool isPalindrome(int x) {
        if (x < 0) return false;

        if (x < 10) return true;

        int len = 0, m = x;
        for (; m !=0; m /= 10) {
            len++;
        }

        int y = x;
        for (int i = 1; i <= len/2; i++) {
            if (x/(int)pow(10, len-i)%10 != y%10) {
                return false;
            }

            y /= 10;
        }

        return true;
    }
};

int main() {
    Solution test;

    cout << test.isPalindrome(1) << endl;
    cout << test.isPalindrome(22) << endl;
    cout << test.isPalindrome(123321) << endl;
    cout << test.isPalindrome(100020001) << endl;
    cout << test.isPalindrome(123) << endl;
    cout << test.isPalindrome(321) << endl;
    cout << test.isPalindrome(12313) << endl;

    return 0;
}
 */
/*
#include <iostream>
#include <string>
#include <limits.h>

using namespace std;

class Solution {
public:
    int myAtoi(string s) {
        int signum = 1;

        int n = 0;
        for (; n < s.size(); n++) {
            if ( s[n] == '-') {
                signum = -1;
                n++;
                break;
            } else if (s[n] == '+') {
                signum = 1;
                n++;
                break;
            } else if ('0' <= s[n] && s[n] <= '9') {
                break;
            } else if (s[n] != ' ') {
                return 0;
            } 
        } 

        long rtn = 0;
        for (;n < s.size(); n++) {
            if ('0' <= s[n] && s[n] <= '9') {
                rtn = rtn*10 + (s[n] - '0');

                if (rtn*signum  < INT_MIN) {
                   return INT_MIN;
                } else if (rtn*signum > INT_MAX) {
                   return INT_MAX;
                }
            } else {
                break;
            }
        }

       return int(rtn*signum);
    }
};

int main() {
    Solution test;

    cout << test.myAtoi("42") << endl;
    cout << test.myAtoi(" -42") << endl;
    cout << test.myAtoi("+-42") << endl;
    cout << test.myAtoi("+-12") << endl;
    cout << test.myAtoi("4193 with words") << endl;
    cout << test.myAtoi("words and 987") << endl;
    cout << test.myAtoi("-91283472332") << endl;

    return 0;
}
 */
/* 
#include <iostream>
#include <limits.h>
#include <string>

using namespace std;

class Solution {
public:
    int reverse(int x) {
        long rtn = 0;

        for (; x != 0; x /= 10) {
            rtn = rtn*10 + x%10;
        } 

        return INT_MIN <= rtn && rtn <= INT_MAX ? rtn : 0;
    }
};

int main() {
    Solution test;

    cout << test.reverse(1) << endl;
    cout << test.reverse(-12) << endl;
    cout << test.reverse(123) << endl;
    cout << test.reverse(1534236469) << endl;
    cout << INT_MIN << endl;
    cout << INT_MAX << endl;

    string tests =  "sfsfs";

    return 0;
}
 */
/* //6
#include <iostream>
#include <string>
#include <vector>

using namespace std;

class Solution {
public:
    string convert(string s, int numRows) {
        if (s.empty() || numRows < 2) return s;

        vector<string> cache(numRows);
        for (int i =0; i < s.size(); i++) {
            int row = i%(numRows-1);
            int col = i/(numRows-1);

            if (col%2 == 0) {
                cache[row] += s[i];
            } else {
                cache[numRows-1-row] += s[i];
            }
        }

        string rtn;
        for (int i =0; i < numRows; i++) {
            rtn += cache[i];
        }

        return rtn;
    }
};	

int main() {
    Solution test;

    cout << (2^2) << end;
    cout << test.convert("a",3) << endl;
    cout << test.convert("vsvs",1) << endl;


    cout << test.convert("PAYPALISHIRING",3) << endl;
    cout << (test.convert("PAYPALISHIRING",3) == "PAHNAPLSIIGYIR") << endl;

    cout << test.convert("PAYPALISHIRING",4) << endl;
    cout << (test.convert("PAYPALISHIRING",4) == "PINALSIGYAHRPI") << endl;

    return 0;
}
 */

/* 
#include <iostream>
#include <string>

using namespace std;

class Solution {
public:
    string longestPalindrome(string s) {
        if (s.size() < 2 ) return s;
        int start = 0, len = 1;

        for (int i = 0; i < s.size(); i++) {
            int left = i-1, right = i+1;
            while (0 <= left && right < s.size() && s[left] == s[right]) {
                if (right - left + 1 > len) {
                    start = left;
                    len = right - left + 1;
                }

                left--;
                right++;
            }

            left = i, right = i+1;
             while (0 <= left && right < s.size() && s[left] == s[right]) {
                if (right - left + 1 > len) {
                    start = left;
                    len = right - left + 1;
                }

                left--;
                right++;
            }
        }

        return s.substr(start, len);
    }
};

int main() {
    Solution test;

    cout << test.longestPalindrome("") << endl;
    cout << test.longestPalindrome(" ") << endl;
    cout << test.longestPalindrome("babad") << endl;
    cout << test.longestPalindrome("cbbd") << endl;
    cout << test.longestPalindrome("a") << endl;
    cout << test.longestPalindrome("ac") << endl;

    return 0;
}
 */
/* 
#include <iostream>
#include <vector>

using namespace std;

class Solution {
public:
    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
        int sizeTotal = nums1.size() + nums2.size();
        int posMid = sizeTotal / 2;
        if (sizeTotal%2==0) {
            posMid--;
        }

        int it1=0, it2 = 0, num1 =0, num2 = 0;
        for (;it1 + it2 <= posMid;) {
            if (it2 == nums2.size()) {
                num1 = nums1[it1++];
            } else if (it1 == nums1.size()) {
                num1 = nums2[it2++];
            } else if (nums1[it1] <= nums2[it2]) {
                num1 = nums1[it1++];
            } else {
                num1 = nums2[it2++];
            }
        }

        double midNum = 1.0;
        if (sizeTotal%2==0) {
            if (it2 == nums2.size() || (it1 < nums1.size() && nums1[it1] <= nums2[it2])){
                num2 = nums1[it1];
            } else {
                num2 = nums2[it2];
            }

            midNum++;
        }

        return ((double)num1 + (double)num2)/midNum;

    }
};

int main(){
    int g[10] = {-1, 0, 1, 2, 3, 4, 5,6, 7, 8};
   
   
    Solution test;
    vector<int> a(g,g+1);
    vector<int> b(g+2,g+2);
  

    cout << test.findMedianSortedArrays(a,b);
    return 0;
}
 */

/* 
#include <iostream> 
#include <set> 
#include <string>

using namespace std;

class Solution {
public:
    int lengthOfLongestSubstring1(string s) {
        int maxLen = 0;
        set<char> cache;
        for (int i=0, j = 0, count = 0; j < s.size(); j++) {
            for (; cache.find(s[j]) != cache.end(); i++) {
                cache.erase(s[i]);
                count--;
            }

            cache.insert(s[j]);
            count++;

            maxLen = maxLen < count ? count : maxLen;
        }

        return maxLen;
    }

     int lengthOfLongestSubstring(string s) {
        int maxLen = 0;

        for (int i=0, j = 0, k = 0; j < s.size(); j++) {
            for (i = k; i < j; i++) {
                if (s[i] == s[j]) {
                    k = i + 1;

                    break;
                }
            }

            maxLen = maxLen < j-k+1 ? j-k+1 : maxLen;
        }

        return maxLen;
    }
};

int main (){
    Solution test;
    cout << test.lengthOfLongestSubstring("") << endl;
    cout << test.lengthOfLongestSubstring(" ") << endl;
    cout << test.lengthOfLongestSubstring("AB") << endl;
    cout << test.lengthOfLongestSubstring("abcabcbb") << endl;
    cout << test.lengthOfLongestSubstring("bbbbb") << endl;
    cout << test.lengthOfLongestSubstring("pwwkew") << endl;
    cout << test.lengthOfLongestSubstring("dvdf") << endl;

    return 0;
}
 */

/* 
#include <iostream>
#include <vector>
#include <map>



//Definition for singly-linked list.
 struct ListNode {
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};
 
class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        if (l1 == NULL && l2 == NULL) {
            return NULL;
        } else if (l1 == NULL) {
            return l1;
        } else if (l2 == NULL) {
            return l2;
        }

        int eAdd = 0;               //记录进位
        ListNode* rtn = NULL;       //记录返回链表头
        ListNode* rtnEnd = NULL;    //记录返回链表尾部，用于新节点插入


        ListNode* it1 = l1;
        ListNode* it2 = l2;

        while (it1 != NULL || it2 != NULL) {
            int newVal = eAdd;
            if (it1 != NULL) {
                newVal += it1->val;
                it1 = it1->next;
            }
            if (it2 != NULL) {
                newVal += it2->val;
                it2 = it2->next; 
            }

            ListNode* newNode = new ListNode();
            newNode->val = newVal%10;
            if (rtnEnd != NULL) {
                rtnEnd->next = newNode;
            } else {
                rtn = newNode;
            }
            rtnEnd = newNode;

            eAdd = newVal/10;
        }

        //边界情况，最后的两个数只和也进位 如：{3}, {9}
        if (eAdd != 0) {
            ListNode* newNode = new ListNode();
            newNode->val = eAdd;

            rtnEnd->next = newNode;
        }

        return rtn;
    }
};

int main() {
    ListNode* l1 = new ListNode();
    l1->val = 3;
    ListNode* l2 = new ListNode();
    l2->val = 9;
    
    Solution test;
    test.addTwoNumbers(l1, l2);

    return 0;
}
 */
/* 
using namespace std;

class solution {
public:
    vector<int> myTest(vector<int> nums, int target) {
        map<int, int> cache;
        vector<int> rtn;

        for (int i= 0; i < nums.size(); i++ ){
            if (cache.find(target -nums[i]) != cache.end()) {
                rtn.push_back(cache[target -nums[i]]);
                rtn.push_back(i);

               break;
            } else {
                cache[nums[i]] = i; 
            }
        }

        return rtn;
    }
};

int main () {
    vector<int> test{3,5,2,1};
    int target = 5;

    solution t;
    vector<int> rtn = t.myTest(test, target);

    for (size_t i =0; i < rtn.size(); i++) {
        cout <<  rtn[i] << " ";
    }

    cout << endl;
    return 0;
} */