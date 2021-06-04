package main

import (
	"fmt"
	"math"
)

func maxSubArray(nums []int) int {
	max, res := float64(nums[0]), float64(nums[0])

	for i := 1; i < len(nums); i++ {
		max = math.Max(float64(nums[i]), float64(nums[i])+max)

		res = math.Max(res, max)
	}

	return int(res)
}

func main() {
	fmt.Println(maxSubArray([]int{1, 2, -3, 1}))
	fmt.Println(maxSubArray([]int{-1, -2, -3, 1}))
	fmt.Println(maxSubArray([]int{-1, -2, -3}))
}

/*
func maxProduct(nums []int) int {
	maxF, minF, res := float64(nums[0]), float64(nums[0]), float64(nums[0])
	for i := 1; i < len(nums); i++ {
		mx, mn := maxF, minF

		maxF = math.Max(mx*float64(nums[i]), math.Max(float64(nums[i]), float64(nums[i])*mn))
		minF = math.Min(mn*float64(nums[i]), math.Min(float64(nums[i]), float64(nums[i])*mx))

		res = math.Max(maxF, res)
	}

	return int(res)
}

func main() {
	// fmt.Println(maxProduct([]int{2, 3, -2, 4, -1}))
	// fmt.Println(maxProduct([]int{-1, -2, -9, -6}))
	fmt.Println(maxProduct([]int{2, -2, -9, -6}))
	// fmt.Println(maxProduct([]int{1, -2}))

}
*/
/*
type ListNode struct {
	Val  int
	Next *ListNode
}

func sortList(head *ListNode) *ListNode {
	cur, l := head, 0
	for cur != nil {
		l++
		cur = cur.Next
	}

	dummyHead := &ListNode{Next: head}

	for j := 1; j < l; j <<= 1 {
		pre, cur := dummyHead, dummyHead.Next
		for cur != nil {
			head1 := cur

			for i := 1; i < j && cur.Next != nil; i++ {
				cur = cur.Next
			}

			head2 := cur.Next
			cur.Next = nil
			cur = head2
			for i := 1; i < j && cur != nil && cur.Next != nil; i++ {
				cur = cur.Next
			}

			var next *ListNode
			if cur != nil {
				next = cur.Next
				cur.Next = nil
			}

			m := merger(head1, head2)
			pre.Next = m

			for pre.Next != nil {
				pre = pre.Next
			}

			cur = next
		}
	}

	return dummyHead.Next
}

func merger(head1, head2 *ListNode) *ListNode {
	dummyHead := &ListNode{}

	now := dummyHead
	for head1 != nil && head2 != nil {
		if head1.Val <= head2.Val {
			now.Next = head1
			head1 = head1.Next
		} else {
			now.Next = head2
			head2 = head2.Next
		}

		now = now.Next
	}

	if head1 != nil {
		now.Next = head1
	}

	if head2 != nil {
		now.Next = head2
	}

	return dummyHead.Next
}

func main() {
	test2 := &ListNode{Val: 1}
	test := &ListNode{Val: 2, Next: test2}
	head := &ListNode{Val: 3, Next: test}

	head = sortList(head)
	for head != nil {
		fmt.Print(head.Val, ", ")
		head = head.Next
	}
}
*/
/*
func longestConsecutive(nums []int) int {
	if len(nums) < 2 {
		return len(nums)
	}

	for _, v := range nums {
		left := cache[v-1]
		right := cache[v+1]

		now := left + right + 1

		if now > max {
			max = now
		}


	}

	return max
}

func longestConsecutive1(nums []int) int {
	if len(nums) < 2 {
		return len(nums)
	}

	max, now := 1, 1

	sort.Ints(nums)

	for i := 1; i < len(nums); i++ {
		if nums[i] == nums[i-1] {
			continue
		} else if nums[i] != nums[i-1]+1 {
			now = 1
		} else {
			now++
			if now > max {
				max = now
			}
		}
	}

	return max
}

func main() {
	fmt.Println(longestConsecutive([]int{1, 2, 0, 1}))
}
*/
/**
解题思路：
按照题意：一颗三个节点的小树的结果只可能有如下6种情况：
根 + 左 + 右
根 + 左
根 + 右
根
左  左右的节点是无意义的，无法越过根
右
然后使用递归，选择小树的最大路径和的情况，拼凑成一颗大树
**/
/*
type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func maxPathSum(root *TreeNode) int {
	max := math.MinInt32

	var bt func(root *TreeNode) int
	bt = func(root *TreeNode) int {
		if root == nil {
			return 0
		}

		left := bt(root.Left)
		if left < 0 {
			left = 0
		}

		right := bt(root.Right)
		if right < 0 {
			right = 0
		}

		if max < left+right+root.Val {
			max = left + right + root.Val
		}

		if left > right {
			return left + root.Val
		}

		return right + root.Val
	}

	bt(root)

	return max
}
*/
/*
func maxProfit(prices []int) int {
	res, dp := 0, 0

	for i := 1; i < len(prices); i++ {
		dp += prices[i] - prices[i-1]

		if dp < 0 {
			dp = 0
		} else if dp > res {
			res = dp
		}
	}

	return res
}

func main() {
	fmt.Println(maxProfit([]int{7, 1, 5, 3, 6, 4}))
	fmt.Println(maxProfit([]int{1, 5, 3}))
	fmt.Println(maxProfit([]int{1, 0, 3}))
	fmt.Println(maxProfit([]int{1, 2, 3}))
	fmt.Println(maxProfit([]int{3, 2, 3}))
	fmt.Println(maxProfit([]int{3, 2, 1}))
}
*/
/*
type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func buildTree(preorder []int, inorder []int) *TreeNode {
	if len(preorder) == 0 {
		return nil
	}

	head := &TreeNode{Val: preorder[0]}

	i := 0
	for k, v := range inorder {
		if v == preorder[0] {
			i = k
			break
		}
	}
	head.Left = buildTree(preorder[1:i+1], inorder[:i])
	head.Right = buildTree(preorder[i+1:], inorder[i+1:])

	return head
}
*/
/*
type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func maxDepth(root *TreeNode) int {
	max, count := 0, 0

	if root == nil {
		return max
	}

	var bt func(root *TreeNode)
	bt = func(root *TreeNode) {
		if root == nil {
			return
		}
		count++
		if count > max {
			max = count
		}
		bt(root.Left)
		bt(root.Right)
		count--
	}

	bt(root)

	return max
}
*/
/*
type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func isSymmetric(root *TreeNode) bool {
	if root == nil {
		return true
	}

	return isSymmetricIn(root.Left, root.Right)
}

func isSymmetricIn(n1, n2 *TreeNode) bool {
	if (n1 == nil && n2 != nil) || (n1 != nil && n2 == nil) {
		return false
	} else if n1 == nil && n2 == nil {
		return true
	} else {
		if n1.Val != n2.Val {
			return false
		}

		return isSymmetricIn(n1.Left, n2.Right) && isSymmetricIn(n1.Right, n2.Left)
	}
}
*/
/*
type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

var preMin = math.MinInt32

func isValidBST1(root *TreeNode) bool {
	if root == nil {
		return true
	}

	if (root.Left != nil && root.Val <= root.Left.Val) || (root.Right != nil && root.Val >= root.Right.Val) {
		return false
	}

	return isValidBST(root.Left) && isValidBST(root.Right)
}

func isValidBST(root *TreeNode) bool {
	if root == nil {
		return true
	}

	isCorrect := isValidBST(root.Left)
	if preMin >= root.Val {
		isCorrect = false
	}

	preMin = root.Val

	if isCorrect {
		isCorrect = isValidBST(root.Right)
	}

	return isCorrect
}

func main() {
	test := &TreeNode{Val: 0}

	fmt.Println(isValidBST(test))
}
*/
//假设n个节点存在二叉排序树的个数是G(n)，令f(i)为以i为根的二叉搜索树的个数
// 即有:G(n) = f(1) + f(2) + f(3) + f(4) + ... + f(n)
// n为根节点，当i为根节点时，其左子树节点个数为[1,2,3,...,i-1]，右子树节点个数为[i+1,i+2,...n]，
// 所以当i为根节点时，其左子树节点个数为i-1个，右子树节点为n-i，即f(i) = G(i-1)*G(n-i),
// 上面两式可得:G(n) = G(0)*G(n-1)+G(1)*(n-2)+...+G(n-1)*G(0)
/*
func numTrees(n int) int {
	dp := make([]int, n+1)
	dp[0] = 1

	for i := 1; i <= n; i++ {
		for j := 0; j < i; j++ {
			dp[i] += dp[j] * dp[i-j-1]
		}
	}

	return dp[n]
}

func main() {
	fmt.Println(numTrees(1))
	fmt.Println(numTrees(2))
	fmt.Println(numTrees(3))
	fmt.Println(numTrees(10))
}
*/
/*
type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func inorderTraversal(root *TreeNode) []int {
	res := make([]int, 0)

	var dp func(root *TreeNode)
	dp = func(root *TreeNode) {
		if root == nil {
			return
		}

		dp(root.Left)
		res = append(res, root.Val)
		dp(root.Right)
	}

	dp(root)

	return res
}
*/

/*
func maximalRectangle(matrix [][]byte) int {
	if len(matrix) == 0 {
		return 0
	}

	dp, ans := make([]int, len(matrix[0])), 0
	for _, v1 := range matrix {
		w := 0

		for j, v2 := range v1 {
			if v2 == '1' {
				w += 1
				dp[j] += 1

				h := math.MaxInt32
				for k := j; k >= 0 && v1[k] != 0; k-- {
					if h > dp[k] {
						h = dp[k]
					}

					if ans < h*(j-k+1) {
						ans = h * (j - k + 1)
					}
				}
			}
		}

	}

	return ans
}

func main() {
	fmt.Println(maximalRectangle([][]byte{{'1', '0', '1', '0', '0'}, {'1', '0', '1', '1', '1'}, {'1', '1', '1', '1', '1'}, {'1', '0', '0', '1', '0'}}))
	// fmt.Println(maximalRectangle([][]byte{{'1', '0', '1', '1', '1'}}))
}
*/
/*
func largestRectangleArea(heights []int) int {
	heights = append([]int{0}, heights...)
	heights = append(heights, 0)

	stack, max := make([]int, 0), 0

	for k, v := range heights {
		for len(stack) != 0 && heights[stack[len(stack)-1]] > v {
			h := heights[stack[len(stack)-1]]
			stack = stack[:len(stack)-1]

			w := k - (stack[len(stack)-1] + 1)

			if max < w*h {
				max = w * h
			}
		}

		stack = append(stack, k)
	}

	return max
}

func main() {
	// fmt.Println(largestRectangleArea([]int{2, 1, 2}))
	fmt.Println(largestRectangleArea([]int{2, 2, 2}))
	fmt.Println(largestRectangleArea([]int{2, 1, 5, 6, 2, 3}))
	fmt.Println(largestRectangleArea([]int{0}))
	fmt.Println(largestRectangleArea([]int{1}))
}
*/
/*
func subsets(nums []int) [][]int {
	res := make([][]int, 0)
	res = append(res, []int{})

	for _, v := range nums {
		for _, vRes := range res {
			tmp := make([]int, len(vRes))
			copy(tmp, vRes)
			tmp = append(tmp, v)

			res = append(res, tmp)
		}
	}

	return res
}
func subsets1(nums []int) [][]int {
	res := make([][]int, 0)
	res = append(res, []int{})

	for _, v := range nums {
		for _, vRes := range res {
			vRes = append(vRes, v)

			res = append(res, vRes)
		}
	}

	return res
}

func subsets2(nums []int) []int {
	for _, v := range nums {
		v *= 2
	}

	return nums
}

type User struct {
	Name string
}

func main() {
	// fmt.Println(subsets([]int{9, 0, 3, 5, 7}))
	fmt.Println(subsets1([]int{9, 0, 3, 5, 7}))
	// fmt.Println(subsets2([]int{9, 0, 3, 5, 7}))
	// fmt.Println(subsets([]int{9, 0, 3, 5}))
}
*/
//上台阶这个可题规律结论是非波那锲数列
//即d[n] = d[n-1] + d[n-2]
/*
func climbStairs(n int) int {
	if n < 3 {
		return n
	}

	p1, p2 := 1, 2
	for i := 3; i <= n; i++ {
		p2, p1 = p1+p2, p2
	}

	return p2
}

func main() {
	fmt.Println(climbStairs(3))
	fmt.Println(climbStairs(4))
	fmt.Println(climbStairs(4))
	fmt.Println(climbStairs(5))
	fmt.Println(climbStairs(6))
}
*/
/*
//思路同下面的多少路径问题
//dp[i]表示当前行，到达i点的最短数量的和， 这个i点只有头上或者左侧过来，去两个较小的在加上当前的位置

func minPathSum(grid [][]int) int {
	dp := make([]int, 0)
	dp = append(dp, grid[0]...)

	//计算第一组的值
	for i := 1; i < len(dp); i++ {
		dp[i] += dp[i-1]
	}

	for i := 1; i < len(grid); i++ {
		for j := 0; j < len(grid[i]); j++ {
			if j != 0 { //头上和左侧较小值+当前值，dp[j]未赋值前表示grid[i][j]头上的最小值
				dp[j] = grid[i][j] + int(math.Min(float64(dp[j]), float64(dp[j-1])))
			} else { //j==0 当前值就是头上值+当前
				dp[j] = grid[i][j] + dp[j]
			}
		}
	}

	return dp[len(grid[0])-1]
}

func main() {
	fmt.Println(minPathSum([][]int{{1, 2, 3}, {4, 5, 6}}))
	fmt.Println(minPathSum([][]int{{1, 2, 3}}))
}
*/
/*
//dp优化
//每个点的dp[i][j] = dp[i-1][j] + dp[i][j-1] 是它头上点和左边点的路径之和

func uniquePaths(m int, n int) int {
	dp := make([]int, n)
	for k := range dp {
		dp[k] = 1
	}

	for j := 1; j < m; j++ {
		for i := 1; i < n; i++ {
			dp[i] += dp[i-1] //被赋值前的dp[i]还是之j-1的i
		}
	}

	return dp[n-1]
}

func uniquePaths2(m int, n int) int {
	res, resF := 1, 1

	for i, j := n-1, 0; i >= 1; i, j = i-1, j+1 {
		res = res * (m + n - 2 - j)
		resF *= i
	}

	return res / resF
}

func uniquePaths1(m int, n int) int {
	num := 0

	var dfs func(m int, n int)
	dfs = func(m int, n int) {
		if m == 0 && n == 0 {
			num++
		}

		if m > 0 {
			dfs(m-1, n)
		}

		if n > 0 {
			dfs(m, n-1)
		}

	}
	dfs(m-1, n-1)

	return num
}

func main() {
	fmt.Println(uniquePaths(3, 3))
	fmt.Println(uniquePaths(2, 2))
	fmt.Println(uniquePaths(7, 3))
	// fmt.Println(uniquePaths(1, 1))
	// fmt.Println(uniquePaths(1, 2))
	// fmt.Println(uniquePaths1(1, 2))
}
*/
/*
func merge(intervals [][]int) [][]int {
	res, start, end := make([][]int, 0), make([]int, 0), make([]int, 0)

	for _, v := range intervals {
		start = append(start, v[0])
		end = append(end, v[1])
	}

	sort.Ints(start)
	sort.Ints(end)

	for i, j := 0, 0; i < len(intervals); i++ {
		if i == len(intervals)-1 || start[i+1] > end[i] {
			res = append(res, []int{start[j], end[i]})
			j = i + 1
		}
	}

	return res
}

func main() {
	fmt.Println(merge([][]int{{1, 4}, {4, 5}}))
} */

/*
func canJump(nums []int) bool {
	can_arrived := 0

	for k, v := range nums {
		if k > can_arrived {
			return false
		}

		if k+v > can_arrived {
			can_arrived = k + v
		}
	}

	return true
}

func main() {
	fmt.Println(canJump([]int{1, 0}))
	fmt.Println(canJump([]int{2, 0}))
	fmt.Println(canJump([]int{1}))
	fmt.Println(canJump([]int{3, 2, 1, 0, 4}))
}
*/

/*
func canJump(nums []int) bool {
	seen := make([]int, len(nums))

	var dfs func(pos int) bool
	dfs = func(pos int) bool {
		if pos >= len(nums)-1 {
			return true
		}

		if seen[pos] == 1 || nums[pos] == 0 && pos < len(nums)-1 {
			return false
		}

		seen[pos] = 1

		for i := nums[pos]; i > 0; i-- {
			if dfs(pos + i) {
				return true
			}
		}

		return false
	}

	return dfs(0)
}

func main() {
	fmt.Println(canJump([]int{1, 0}))
	fmt.Println(canJump([]int{2, 0}))
	fmt.Println(canJump([]int{1}))
	fmt.Println(canJump([]int{3, 2, 1, 0, 4}))
}
*/
/*
//优化dp
func maxSubArray(nums []int) int {
	sum, res := 0, nums[0]

	for _, v := range nums {
		// if sum + v > v { //前面总和加当前大于之前，就加上去
		if sum > 0 { //如果是正数，必增加
			sum += v
		} else { //如果负数，前面和是-100,当前-1，把-1替换成新的开头
			sum = v
		}

		if sum > res {
			res = sum
		}
	}

	return res
}

// 思考状态：dp[i]代表着以nums[i]结尾的最大的子序列和。
// 思考状态转移方程：
// dp[i] = Math.max(dp[i-1] + nums[i], nums[i]);
// 取dp[i-1]+nums[i]和nums[i]的最大值是因为考虑dp[i-1]是否对nums[i]产生了负增益，如果对nums[i]产生了负增益，那么不如不产生，对应的就是将dp[i-1]去掉。
// 思考初始化：dp[0] = nums[0]，所以i必须从1开始直到末尾。
// 思考输出：输出dp数组的最大值即可。
func maxSubArray1(nums []int) int {
	dp := make([]int, len(nums))
	dp[0] = nums[0]

	for i := 1; i < len(nums); i++ {
		dp[i] = int(math.Max(float64(dp[i-1]+nums[i]), float64(nums[i])))
	}

	max := dp[0]
	for _, v := range dp {
		if v > max {
			max = v
		}
	}

	return max
}

func main() {
	// fmt.Println(maxSubArray([]int{-2, 1, -3, 4, -1, 2, 1, -5, 4}))
	// fmt.Println(maxSubArray([]int{-2, 1}))
	fmt.Println(maxSubArray1([]int{-2, 1, -3, 4, -1, 2, 1, -5, 4}))
	fmt.Println(maxSubArray1([]int{-2, 1}))
}
*/
/*
type SortBy []byte

func (a SortBy) Len() int           { return len(a) }
func (a SortBy) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a SortBy) Less(i, j int) bool { return a[i] < a[j] }

func myBtoA(t SortBy) string {
	res := ""
	for _, v := range t {
		res += string(v)
	}

	return res
}

func groupAnagrams(strs []string) [][]string {
	cache, res := make(map[string][]string, 0), make([][]string, 0)

	for _, v := range strs {
		p := SortBy(v)
		sort.Sort(p)
		key := myBtoA(p)

		cache[key] = append(cache[key], v)
	}

	for _, v := range cache {
		res = append(res, v)
	}

	return res
}

func main() {
	fmt.Println(groupAnagrams([]string{"eat", "tea", "tan", "ate", "nat", "bat"}))
	v := "eat"
	// p := SortBy(v)

	sort.Sort(SortBy(v))
	fmt.Println("==============")
	// fmt.Println(myBtoA(p))
	fmt.Println(v)

}
*/
/*
func rotate(matrix [][]int) {
	for i := 0; i < len(matrix)/2; i++ {
		matrix[i], matrix[len(matrix)-1-i] = matrix[len(matrix)-1-i], matrix[i]
	}

	for i := 0; i < len(matrix); i++ {
		for j := 0; j < len(matrix[i]); j++ {
			if i < j {
				matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
			}
		}
	}
} */

/*
func trap(height []int) int {
	sum, maxLeft, maxRight, left, right := 0, 0, 0, 1, len(height)-2

	for i := 1; i < len(height)-1; i++ {
		if height[left-1] < height[right+1] {
			maxLeft = int(math.Max(float64(maxLeft), float64(height[left-1])))

			if maxLeft > height[left] {
				sum = sum + (maxLeft - height[left])
			}

			left++
		} else {
			maxRight = int(math.Max(float64(maxRight), float64(height[right+1])))

			if maxRight > height[right] {
				sum = sum + (maxRight - height[right])
			}

			right--
		}
	}

	return sum
}

func main() {
	fmt.Println(trap([]int{4, 1, 0, 3, 2}))
	// fmt.Println(trap([]int{4, 0, 2, 0, 3, 2}))
}
*/
/*
func permute(nums []int) [][]int {
	res, seen := make([][]int, 0), make([]int, len(nums))

	var dfs func(nums, ans []int)
	dfs = func(nums, ans []int) {
		if len(ans) == len(nums) {
			ansCopy := make([]int, len(ans))
			copy(ansCopy, ans) //如果不用copy，后面还是用到这个切片会覆盖掉当前值
			res = append(res, ansCopy)
			return
		}

		for k, v := range nums {
			if seen[k] == 1 { //以前遍历过的，不用再遍历，负责出现重复的解
				continue
			}

			seen[k] = 1
			ans = append(ans, v)
			dfs(nums, ans)
			ans = ans[:len(ans)-1]
			seen[k] = 0
		}
	}

	dfs(nums, []int{})

	return res
}

func main() {
	fmt.Println(permute([]int{1, 2, 3}))
}
*/
/*
1. 将target-candidates[i]值减去，就得到一个新的target
*/
/*
func combinationSum(candidates []int, target int) [][]int {
	res := make([][]int, 0)

	var dfs func(candidates []int, target int, ans []int)
	dfs = func(candidates []int, target int, ans []int) {
		if target == 0 {
			ansCopy := make([]int, len(ans))
			copy(ansCopy, ans) //如果不用copy，后面还是用到这个切片
			res = append(res, ansCopy)
			return
		}

		for k, v := range candidates {
			if target < v {
				continue
			}

			ans = append(ans, v)
			dfs(candidates[k:], target-v, ans) //包括k（k可以重复加），以前遍历过的，不用再遍历，负责出现重复的解
			ans = ans[:len(ans)-1]
		}
	}

	dfs(candidates, target, []int{})

	return res
}

func main() {
	// fmt.Println(combinationSum([]int{2, 3}, 2))
	fmt.Println(combinationSum([]int{2, 3, 6, 7}, 7))
}
*/
/*
题目要求：o(logn)复杂度
1.意思就是用二分查找
2. 所以遍历两次，一次找到最左边，一次找到最右边的
*/
/*
func searchRange(nums []int, target int) []int {
	pos1, pos2, begin, end := -1, -1, 0, len(nums)-1

	for begin <= end {
		mid := (begin + end) >> 1
		// test := begin + (end-begin)>>1

		if nums[mid] == target {
			pos1 = mid
			end = mid - 1 //继续二分查找最左边的，至begin > end
		} else if nums[mid] < target {
			begin = mid + 1
		} else {
			end = mid - 1
		}
	}

	begin, end = 0, len(nums)-1
	for begin <= end {
		mid := (begin + end) >> 1

		if nums[mid] == target {
			pos2 = mid
			begin = mid + 1 //继续二分查找最右边的
		} else if nums[mid] < target {
			begin = mid + 1
		} else {
			end = mid - 1
		}
	}

	return []int{pos1, pos2}
}

func main() {
	fmt.Println(searchRange([]int{5, 7, 7, 8, 8, 10}, 4))
	fmt.Println(searchRange([]int{5, 7, 7, 8, 8, 10}, 7))
	fmt.Println(searchRange([]int{7, 7, 7, 8, 8, 8}, 7))
	fmt.Println(searchRange([]int{7, 7, 7, 8, 8, 8}, 8))
}
*/

/*
//[4,5,6,7,0,1,2]
func search(nums []int, target int) int {
	begin, end, pos := 0, len(nums)-1, -1

	for begin <= end {
		mid := (begin + end) >> 1

		if nums[mid] == target {
			return mid
		} else if nums[mid] < nums[end] {
			if nums[mid] < target && target <= nums[end] {
				begin = mid + 1
			} else {
				end = mid - 1
			}
		} else {
			if nums[begin] <= target && target < nums[mid] {
				end = mid - 1
			} else {
				begin = mid + 1
			}
		}
	}

	return pos
}

func main() {
	fmt.Println(search([]int{4, 5, 6, 7, 0, 1, 2}, 0))
}
*/
/*
动态规划的方法
dp数组dp[i+1] 表示到是s[i]已经匹配的个数
*/
/*
func longestValidParentheses(s string) int {
	//去拿已匹配字符串的长度时，用dp[0] = 0表示，字符0号位置之前的匹配情况，这样做就是为了防止去处理边界校验问题
	//dp[i] 表示s[i-1]为结尾的的字符，已经匹配的个数
	dp := make([]int, len(s)+1)

	for i := 1; i < len(s); i++ {
		if s[i] == ')' {
			if s[i-1] == '(' { //"()"
				dp[i+1] = dp[i-1] + 2
			} else if i-dp[i]-1 >= 0 && s[i-dp[i]-1] == '(' { //eg1:"())"  eg2:"()(())"
				dp[i+1] = dp[i] + 2 + dp[i-dp[i]-1]
			}
		}
	}

	max := 0
	for _, v := range dp {
		if v > max {
			max = v
		}
	}

	return max
}

func main() {
	// fmt.Println(longestValidParentheses("()(()"))
	// fmt.Println(longestValidParentheses("()(())"))
	// fmt.Println(longestValidParentheses("(())"))
	// fmt.Println(longestValidParentheses(")()())"))
	// fmt.Println(longestValidParentheses("()"))
	// fmt.Println(longestValidParentheses("())"))
	fmt.Println(longestValidParentheses("(()())"))

}
*/
/*
1. 从后往前找到一个递减的数字位置为pos，这个数字位置就是需要变动的
2. 找到遍历过的数中刚好大于这个数字替换掉第一步找到的数字
3. 把pos+1到end之间排序，这样使后面这个数值最小
*/
/*
type SortBy []int

func (a SortBy) Len() int           { return len(a) }
func (a SortBy) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a SortBy) Less(i, j int) bool { return a[i] < a[j] }

func nextPermutation(nums []int) {
	l := len(nums)
	if l < 2 {
		return
	}

	pos := -1
	for i := l - 2; i >= 0; i-- {
		if nums[i] < nums[i+1] {
			pos = i

			break
		}
	}

	if pos != -1 {
		for i := l - 1; i >= 0; i-- {
			if nums[i] > nums[pos] {
				nums[i], nums[pos] = nums[pos], nums[i]

				sort.Sort(SortBy(nums[pos+1:]))
				break
			}
		}
	} else {
		for i, j := 0, l-1; i < j; i, j = i+1, j-1 {
			nums[i], nums[j] = nums[j], nums[i]
		}
	}
}

func main() {
	nums := []int{1, 1, 5}
	nextPermutation(nums)

	nums1 := []int{1, 3, 2}
	nums2 := []int{2, 3, 1}
	nums3 := []int{4, 2, 0, 2, 3, 2, 0}
	nums4 := []int{2, 3, 2, 0}
	// nextPermutation(nums1)
	// nextPermutation(nums3)
	// nextPermutation(nums4)

	fmt.Println("-----------")
	fmt.Println(nums1)
	nextPermutation(nums1)
	fmt.Println(nums1)
	fmt.Println("-----------")
	fmt.Println(nums2)
	nextPermutation(nums2)
	fmt.Println(nums2)
	fmt.Println("-----------")
	fmt.Println(nums3)
	nextPermutation(nums3)
	fmt.Println(nums3)
	fmt.Println("-----------")
	fmt.Println(nums4)
	nextPermutation(nums4)
	fmt.Println(nums4)
}
*/
/*
type ListNode struct {
	Val  int
	Next *ListNode
}

func mergeKLists(lists []*ListNode) *ListNode {
	i, min := -1, int(^uint(0)>>1)
	for k, v := range lists {
		if v != nil && v.Val < min {
			min = v.Val
			i = k
		}
	}

	if i == -1 {
		return nil
	}

	newNode := lists[i]
	if lists[i].Next == nil {
		copy(lists[i:], lists[i+1:])
		lists = lists[:len(lists)-1]
	} else {
		lists[i] = lists[i].Next
	}

	newNode.Next = mergeKLists(lists)

	return newNode
}

func main() {
	node1 := &ListNode{Val: 1}
	node2 := &ListNode{Val: 2}
	node3 := &ListNode{Val: 3}
	test := []*ListNode{nil, nil, nil, node1}

	fmt.Println(test)
}
*/
/*
func generateParenthesis(n int) []string {
	res := make([]string, 0)

	var dfs func(l, r int, curStr string)
	dfs = func(l, r int, curStr string) {
		if l > r { //剪枝
			return
		}

		if l == r && l == 0 { //一个解
			res = append(res, curStr)
		}

		if l > 0 { //左 还有继续
			dfs(l-1, r, curStr+"(")
		}

		if r > 0 { // 右 还有继续
			dfs(l, r-1, curStr+")")
		}
	}

	dfs(n, n, "")

	return res
}

func main() {
	fmt.Println(generateParenthesis(4))
	fmt.Println(generateParenthesis(3))
	fmt.Println(generateParenthesis(2))
	fmt.Println(generateParenthesis(1))
}
*/
/*
21. 合并有序链表

*/

/*
//  Definition for singly-linked list.
type ListNode struct {
	Val  int
	Next *ListNode
}

func mergeTwoLists(l1 *ListNode, l2 *ListNode) *ListNode {
	if (l1 == nil && l2 == nil) || (l1 == nil && l2 != nil) {
		return l2
	} else if l1 != nil && l2 == nil {
		return l1
	} else if l1.Val <= l2.Val {
		l1.Next = mergeTwoLists(l1.Next, l2) //利用递归去把下一个节点赋值给新当前的next
		return l1
	} else {
		l2.Next = mergeTwoLists(l1, l2.Next)
		return l2
	}
}
func main() {
	_ = mergeTwoLists(nil, nil)
}
*/
/*
func isValid(s string) bool {
	stack := make([]rune, 0)

	for _, v := range s {
		switch v {
		case '(', '{', '[':
			stack = append(stack, v)
		case ')':
			if len(stack) != 0 && stack[len(stack)-1] == '(' {
				stack = stack[:len(stack)-1]
			} else {
				return false
			}
		case '}':
			if len(stack) != 0 && stack[len(stack)-1] == '{' {
				stack = stack[:len(stack)-1]
			} else {
				return false
			}
		case ']':
			if len(stack) != 0 && stack[len(stack)-1] == '[' {
				stack = stack[:len(stack)-1]
			} else {
				return false
			}
		default:
			return false
		}

	}

	return len(stack) == 0
}

func main() {
	fmt.Println(isValid("()"))
	fmt.Println(isValid("()[]{}"))
	fmt.Println(isValid("{[]}"))
	fmt.Println(isValid("(]"))
}
*/
/**
 * Definition for singly-linked list.
 */
/*
type ListNode struct {
	Val  int
	Next *ListNode
}

func removeNthFromEnd(head *ListNode, n int) *ListNode {

	cur := 0
	var removeNthFromEndIn func(head *ListNode, n int) *ListNode
	removeNthFromEndIn = func(head *ListNode, n int) *ListNode {
		if head == nil {
			return head
		}

		head.Next = removeNthFromEndIn(head.Next, n)
		cur++

		if cur == n {
			return head.Next
		}

		return head
	}

	return removeNthFromEndIn(head, n)
}

func myPrintf(head *ListNode) {
	if head == nil {
		return
	}

	fmt.Println(*head)
	myPrintf(head.Next)
}

func add(cur int) {
	cur++
}

func main() {
	newNode := &ListNode{Val: 2}
	head := &ListNode{Val: 1, Next: newNode}

	myPrintf(head)
	fmt.Println("---------------")
	removeNthFromEnd(head, 1)
	myPrintf(head)
}
*/
/*17. 电话号码的字母组合
1. 深度优先遍历
2.
*/
/*
func letterCombinations(digits string) []string {
	phoneNUm := map[string]string{
		"2": "abc",
		"3": "def",
		"4": "ghi",
		"5": "jkl",
		"6": "mno",
		"7": "pqrs",
		"8": "tuv",
		"9": "wxyz",
	}

	res, cur := make([]string, 0), ""
	if len(digits) == 0 {
		return res
	}

	var dfs func(digits string)
	dfs = func(digits string) {
		n := digits[0]
		letter := phoneNUm[string(n)]

		for _, v := range letter {
			cur += string(v)

			if len(digits) > 1 {
				dfs(digits[1:])
			} else { //到达了一个底，得到一个解
				res = append(res, cur)
			}

			if len(cur) > 1 { //回退
				cur = cur[:len(cur)-1]
			} else {
				cur = ""
			}
		}
	}

	dfs(digits)

	return res
}

func main() {
	// fmt.Println(letterCombinations("2"))
	fmt.Println(letterCombinations("23"))
	fmt.Println(letterCombinations("234"))
}
*/
/*
//三数字之和

import (
	"fmt"
	"sort"
)

type SortBy []int

func (a SortBy) Len() int           { return len(a) }
func (a SortBy) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a SortBy) Less(i, j int) bool { return a[i] < a[j] }

func threeSum(nums []int) [][]int {
	if len(nums) < 3 {
		return nil
	}

	sort.Sort(SortBy(nums))
	rtn := make([][]int, 0)

	for i := 0; i <= len(nums)-3; i++ {
		if i != 0 && nums[i] == nums[i-1] { //去重
			continue
		}

		for j, end := i+1, len(nums)-1; j < end; j++ {
			if j > i+1 && nums[j] == nums[j-1] { //去重
				continue
			}

			for end-1 > j && nums[i]+nums[j]+nums[end] > 0 {
				end--
			}

			if nums[i]+nums[j]+nums[end] == 0 {
				rtn = append(rtn, []int{nums[i], nums[j], nums[end]})
			}

		}
	}

	return rtn
}

func main() {
	fmt.Println(threeSum([]int{-1, 0, 1, 2, -1, -4}))
	fmt.Println(threeSum([]int{0, 0}))
	fmt.Println(threeSum([]int{0, 0, 0}))
	fmt.Println(threeSum([]int{0, 0, 0, 0}))
	fmt.Println(threeSum([]int{-12, 4, 12, -4, 3, 2, -3, 14, -14, 3, -12, -7, 2, 14, -11, 3, -6, 6, 4, -2, -7, 8, 8, 10, 1, 3, 10, -9, 8, 5, 11, 3, -6, 0, 6, 12, -13, -11, 12, 10, -1, -15, -12, -14, 6, -15, -3, -14, 6, 8, -9, 6, 1, 7, 1, 10, -5, -4, -14, -12, -14, 4, -2, -5, -11, -10, -7, 14, -6, 12, 1, 8, 4, 5, 1, -13, -3, 5, 10, 10, -1, -13, 1, -15, 9, -13, 2, 11, -2, 3, 6, -9, 14, -11, 1, 11, -6, 1, 10, 3, -10, -4, -12, 9, 8, -3, 12, 12, -13, 7, 7, 1, 1, -7, -6, -13, -13, 11, 13, -8}))
}
*/
/*
// 12
func intToRoman(num int) string {
	var tag = map[int]string{
		1000: "M",
		900:  "CM",
		500:  "D",
		400:  "CD",
		100:  "C",
		90:   "XC",
		50:   "L",
		40:   "XL",
		10:   "X",
		9:    "IX",
		5:    "V",
		4:    "IV",
		1:    "I",
	}

	var weight = []int{1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1}

	rtn := ""
	for _, v := range weight {
		for i := num / v; i != 0; i-- {
			rtn += tag[v]
		}

		num %= v
	}

	return rtn
}

func main() {
	fmt.Println(intToRoman(3))
	fmt.Println(intToRoman(4))
	fmt.Println(intToRoman(4))
	fmt.Println(intToRoman(9))
	fmt.Println(intToRoman(11))
	fmt.Println(intToRoman(58))
	fmt.Println(intToRoman(1994))
	fmt.Println(intToRoman(3999))
}
*/
/*
11.
1. 双指针法，移动两个之间较小的数.如果移动叫小的，始终是较小的高乘以缩短的间距，这样这个数只会比上一次小
*/
/*
func maxArea(height []int) int {
	maxArea := 0

	for i, j := 0, len(height)-1; i < j; {

		if height[i] <= height[j] {
			if maxArea < height[i]*(j-i) {
				maxArea = height[i] * (j - i)
			}

			i++
		} else {
			if maxArea < height[j]*(j-i) {
				maxArea = height[j] * (j - i)
			}

			j--
		}
	}

	return maxArea
}

func main() {
	fmt.Println(maxArea([]int{1, 8, 6, 2, 5, 4, 8, 3, 7}))
	fmt.Println(maxArea([]int{1, 1}))
	fmt.Println(maxArea([]int{4, 3, 2, 1, 4}))
	fmt.Println(maxArea([]int{1, 2, 1}))
}
*/
/*
10 正则表达式匹配
dp数组就是去将s字符从第一个字符开始匹配的结果填入到数组，根据不同情况使用可以复用之前的结果
*/
/*
func isMatch(s string, p string) bool {
	// var dp [len(s) + 1][len(p) + 1]bool
	dp := make([][]bool, len(s)+1)

	for k := range dp {
		dp[k] = make([]bool, len(p)+1)
	}

	//p，s都是空
	dp[0][0] = true

	//处理s为空的情况
	for j := 1; j <= len(p); j++ {
		if p[j-1] != '*' {
			continue
		} else {
			dp[0][j] = dp[0][j-2]
		}
	}

	//遍历填写dp数组
	for i := 1; i <= len(s); i++ {
		for j := 1; j <= len(p); j++ {
			if s[i-1] == p[j-1] || p[j-1] == '.' {
				dp[i][j] = dp[i-1][j-1]
			} else if p[j-1] == '*' {
				//"a", "ab*"
				dp[i][j] = dp[i][j-2]

				//判断条件加 p[j-2] == '.'     "ab", ".*",也需要用到dp[i-1][j]
				//"a", "a*"  第二种情况需要用到 dp[i-1][j]
				//"a", "ab*a*"  赋值时不加上dp[i][j]，dp[i-1][j]会覆盖掉前面的dp[i][j]，两者求或 (因为我们尽可能去匹配字符串，所以只要有一种情况匹配成功，则匹配成功)
				if p[j-2] == '.' || p[j-2] == s[i-1] {
					dp[i][j] = dp[i][j] || dp[i-1][j]
				}
			}
		}
	}

	// for _, v := range dp {
	// 	fmt.Println(v)
	// }

	return dp[len(s)][len(p)]
}
*/

/*
//error
func isMatch(s string, p string) bool {
	l1, l2 := len(s), len(p)
	if l1 == 0 || l2 == 0 {
		return (l1 == 0) && (l2 == 0 || (l2 == 2 && p[l2-1] == '*'))
	}

	if p[l2-1] == '.' || s[l1-1] == p[l2-1] {
		return isMatch(s[:l1-1], p[:l2-1])
	} else {
		if p[l2-1] == '*' {
			pos1 := l1 - 1
			pos2 := l2 - 2
			tag := s[pos1]

			if pos2 >= 0 { // "a", "c*"
				if p[pos2] != tag {
					if p[pos2] != '.' {
						return isMatch(s[:l1], p[:pos2])
					} else {
						return isMatch(s[:l1-1], p[:l2-1])
					}
				} else { // "aaa", "aaa*"
					for pos1 >= 0 && tag == s[pos1] {
						pos1--
					}

					for pos2 >= 0 && p[pos2] == tag {
						pos2--
					}

					return isMatch(s[:pos1+1], p[:pos2+1])
				}
			} else {
				return false
			}
		} else {
			return false
		}
	}
}

func main() {
	fmt.Println(isMatch("ab", ".*"))
	fmt.Println(isMatch("aab", "c*a*b"))
	fmt.Println(isMatch("aa", "a*"))
	fmt.Println(isMatch("ab", ".*c"))
	fmt.Println(isMatch("aa", "a"))
	fmt.Println(isMatch("mississippi", "mis*is*p*."))
	fmt.Println(isMatch("aaa", "ab*a"))
	fmt.Println(isMatch("aaa", "ab*a*c*a"))
	fmt.Println(isMatch("aab", "ab*a*c*"))
	fmt.Println(isMatch("aaba", "ab*a*c*a"))
}
*/
/* func isMatch(s string, p string) bool {
	i, j := 0, 0

	for i < len(s) && j < len(p) {
		if s[i] != p[j] {
			if p[j] == '.' {
				i++
				j++
			} else if p[j] == '*' {
				tag := s[i]

				for i < len(s) && tag == s[i] {
					i++
				}

				if i < len(s) {
					j++
				} else {
					break
				}
			} else if j+1 < len(p) && p[j+1] == '*' {
				i++
				j += 2
			} else {
				return false
			}
		} else {
			i++
			j++
		}
	}

	if i < len(s) {
		return false
	}

	return true
} */

/*
9回文数
*/

/*
//1221
func isPalindrome(x int) bool {
	if x < 0 {
		return false
	}

	if x < 10 {
		return true
	}

	m, n := x, 0
	for ; m != 0; m /= 10 {
		n++
	}

	//用y来去最后一个数字
	y := x
	for i := 1; i <= n/2; i++ {
		if x/int(math.Pow10(n-i))%10 != y%10 {
			return false
		}

		y = y / 10
	}

	return true
}

func main() {
	fmt.Println(isPalindrome(-121))

	fmt.Println(isPalindrome(10022201))
	fmt.Println(isPalindrome(-121))
	fmt.Println(isPalindrome(11234456165443211))
	fmt.Println(isPalindrome(123))
	// fmt.Println(isPalindrome(121))
	fmt.Println(isPalindrome(1221))
	fmt.Println(isPalindrome(123321))
	fmt.Println(isPalindrome(12344566544321))
	fmt.Println(isPalindrome(123445616544321))
}
*/
/*
8 myAtoi
*/
/*
func myAtoi(s string) int {
	n := 0
	signum := int32(1)

	for k, v := range s {
		if v == '-' {
			signum *= -1
			n = k + 1

			break
		} else if v == '+' {
			n = k + 1

			break
		} else if '0' <= v && v <= '9' {
			n = k
			break
		} else if v != ' ' {
			return 0
		}
	}

	var rtn int32
	for _, v := range s[n:] {
		if '0' <= v && v <= '9' {
			checkNum := rtn

			rtn = rtn*10 + (v - '0')

			//溢出
			if rtn/10 != checkNum {
				max_int := int(^uint32(0) >> 1)

				if signum == 1 {
					return max_int
				} else {
					return -max_int - 1
				}
			}
		} else {
			break
		}
	}

	return int(signum * rtn)
}

func main() {
	fmt.Println(myAtoi("42"))
	fmt.Println(myAtoi(" -42"))
	fmt.Println(myAtoi("+-12"))
	fmt.Println(myAtoi("4193 with words"))
	fmt.Println(myAtoi("words and 987"))
	fmt.Println(myAtoi("-91283472332"))
}
*/

/* 7.整数反转
 */

/*
func reverse(x int) int {
	y, rtn := int32(x), int32(0)

	for ; y != 0; y /= 10 {
		checkNum := rtn
		rtn = rtn*10 + y%10

		if rtn/10 != checkNum { //乘以10，再除以10不相等了，就是一出了
			rtn = 0

			break
		}
	}

	return int(rtn)
}

func main() {
	fmt.Printf("%d\n", reverse(-2))
	fmt.Printf("%d\n", reverse(-12))
	fmt.Printf("%d\n", reverse(123))
	fmt.Printf("%d\n", reverse(1534236469))

	test2 := int32((^uint32(0) >> 1))
	test3 := -test2 - 1

	test := 0
	fmt.Printf("%d\n", unsafe.Sizeof(test))
	fmt.Printf("%d\n", test2)
	fmt.Printf("%d\n", test3)

}
*/

/*6. Z 字形变换
输入：
"PAYPALISHIRING"  3
输出先排序：
P   A   H   N
A P L S I I G
Y   I   R
结果：
"PAHNAPLSIIGYIR"

这题我觉着主要是找规律的

输出先排序：				*
P   A   H   N			  **		   P   A   H   N
A P L S I I G    *************		   A P L S I I G
Y   I   R				  **		     Y   I   R
						  *
						             偶数列正序放入缓存，偶数列倒序放入缓存

*/

/*
func convert(s string, numRows int) string {
	if s == "" || numRows < 2 {
		return s
	}

	cache := make([]string, numRows)

	for k, v := range s {
		row := k % (numRows - 1)
		line := k / (numRows - 1)

		if line%2 == 0 { //偶数列正序
			cache[row] += string(v)
		} else { //奇数逆序
			cache[numRows-1-row] += string(v)
		}
	}

	rtn := ""
	for _, v := range cache {
		rtn += v
	}

	return rtn
}

func main() {
	fmt.Println("%v", convert("PAYPALISHIRING", 3))
	fmt.Println("%v", convert("PAYPALISHIRING", 3) == "PAHNAPLSIIGYIR")
}
*/
//aba
//abba
/*
五. 最长回文子串
思路：
1.遍历s，把每一个字符串的点作为中心，比较两边是否相等、到达边缘
2.如果相同，记录回文的开始和长度
3.还有这种abba回文是把当前节点和下一个节点作为中心

*/
/*
func longestPalindrome(s string) string {
	if len(s) < 2 {
		return s
	}

	//ab 这种回文的是a或者b长度为1,用中心法则出来是0,因此我们初始长度设置为1
	start, length := 0, 1

	var expendAroundCenter func(left, right int)
	expendAroundCenter = func(left, right int) {
		for left >= 0 && right < len(s) && s[left] == s[right] {
			if right-left+1 > length {
				start = left
				length = right - left + 1
			}

			left--
			right++
		}
	}

	for i := 0; i < len(s); i++ {
		expendAroundCenter(i-1, i+1)
		expendAroundCenter(i, i+1)
	}

	return s[start : start+length]
}

func main() {
	fmt.Println("%v", longestPalindrome(""))
	fmt.Println("%v", longestPalindrome(" "))
	fmt.Println("%v", longestPalindrome("bb"))
	fmt.Println("%v", longestPalindrome("aba"))
	fmt.Println("%v", longestPalindrome("abba"))
	fmt.Println("%v", longestPalindrome("abcbad"))
	fmt.Println("%v", longestPalindrome("dabcba"))
}
*/

/*
四 寻找两个正序数组的中位数
给定两个大小分别为 m 和 n 的正序（从小到大）数组 nums1 和 nums2。请你找出并返回这两个正序数组的 中位数 。
思路：归并排序
1. 计算出两个数组的长度，然后算出中位数的位置
2. 两个指针指向两个数组的下标遍历，找到中位数在那个数组中
*/

/*
func findMedianSortedArrays(nums1 []int, nums2 []int) float64 {
	if len(nums1) == 0 && len(nums2) == 0 {
		return 0
	}

	lenTotal := len(nums1) + len(nums2)
	posMid := lenTotal / 2
	if lenTotal%2 == 0 {
		posMid--
	}
	k1, k2, num1, num2 := 0, 0, 0, 0

	for k1+k2 <= posMid {
		if k2 == len(nums2) {
			num1 = nums1[k1]
			k1++
		} else if k1 == len(nums1) {
			num1 = nums2[k2]
			k2++
		} else if nums1[k1] <= nums2[k2] {
			num1 = nums1[k1]
			k1++
		} else {
			num1 = nums2[k2]
			k2++
		}
	}

	midNum := 1.0
	if lenTotal%2 == 0 {
		if k2 == len(nums2) || (k1 < len(nums1) && nums1[k1] <= nums2[k2]) {
			num2 = nums1[k1]
		} else {
			num2 = nums2[k2]
		}

		midNum++
	}

	return (float64(num1) + float64(num2)) / midNum
}

func main() {
	fmt.Printf("%f\n", findMedianSortedArrays([]int{}, []int{-1, 0, 0, 0, 0, 0, 1}))
	fmt.Printf("%f\n", findMedianSortedArrays([]int{}, []int{}))
	fmt.Printf("%f\n", findMedianSortedArrays([]int{}, []int{1}))
	fmt.Printf("%f\n", findMedianSortedArrays([]int{1}, []int{}))
	fmt.Printf("%f\n", findMedianSortedArrays([]int{1, 2}, []int{}))
	fmt.Printf("%f\n", findMedianSortedArrays([]int{1, 2}, []int{3}))
	fmt.Printf("%f\n", findMedianSortedArrays([]int{1, 2}, []int{3, 4}))
	fmt.Printf("%f\n", findMedianSortedArrays([]int{1, 2}, []int{3, 4, 5}))
	fmt.Printf("%f\n", findMedianSortedArrays([]int{3, 4, 5}, []int{1, 2}))
}
*/

/*
三. 无重复字符的最长子串
思路：1.用三个指针标记，i,j,k
2.j开始遍历，k用记录上一个出现相同的后一个记录，i则作为j每增加一个遍历k到j的之间的是否存在和新增加j相同的字符
3.记录j到k之间的最大个数，就是最后的答案
*/
/*
//dvdf
//"abcabcbb"
func lengthOfLongestSubstring(s string) int {
	maxLen := 0
	// seen := make(map[byte]bool, 0)

	for i, j, k := 0, 0, 0; j < len(s); j++ {
		for i = k; i < j; i++ {
			if s[i] == s[j] {
				k = i + 1
				break
			}
		}

		if maxLen < j-k+1 {
			maxLen = j - k + 1
		}
	}

	return maxLen
}

func main() {
	fmt.Printf("%d\n", lengthOfLongestSubstring(""))
	fmt.Printf("%d\n", lengthOfLongestSubstring("bbbbbbb"))
	fmt.Printf("%d\n", lengthOfLongestSubstring("au"))
	fmt.Printf("%d\n", lengthOfLongestSubstring("dvdf"))
	fmt.Printf("%d\n", lengthOfLongestSubstring("abcabcbb"))
	fmt.Printf("%d\n", lengthOfLongestSubstring("abcd"))
}
*/

/*
这个问题其实bigdata相加问题
1.倒着遍历两个链表
2.需要考虑返回的链表头
3.需要一个数字录每一位进位的值  eAdd = (l1.val + l2.val + eAdd)/10，需要注意的是有可能其中一个链表位数不够了
4.新的一位数字等于 (l1.val + l2.val + eAdd)%10
5.考虑边界问题，最后的一位有进位，但是链表next为空，循环中处理不到
*/
/*
//Definition for singly-linked list.
type ListNode struct {
	Val  int
	Next *ListNode
}

func addTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
	var head, end *ListNode
	it1, it2 := l1, l2
	eAdd := 0

	for it1 != nil || it2 != nil {
		newVal := eAdd
		if it1 != nil {
			newVal += it1.Val
			it1 = it1.Next
		}
		if it2 != nil {
			newVal += it2.Val
			it2 = it2.Next
		}

		newNode := &ListNode{Val: newVal % 10}

		if end != nil {
			end.Next = newNode
		} else {
			head = newNode
		}
		end = newNode

		eAdd = newVal / 10
	}

	if eAdd != 0 {
		newNode := &ListNode{Val: eAdd}
		end.Next = newNode
	}

	return head
}

func main() {
	var l1, l2, end *ListNode
	for i := 0; i < 7; i++ {
		newNode := &ListNode{Val: 9}
		if end != nil {
			end.Next = newNode
		} else {
			l1 = newNode
		}
		end = newNode
	}

	end = nil
	for i := 0; i < 4; i++ {
		newNode := &ListNode{Val: 9}
		if end != nil {
			end.Next = newNode
		} else {
			l2 = newNode
		}
		end = newNode
	}

	l3 := addTwoNumbers(l1, l2)
	for it := l3; it != nil; it = it.Next {
		fmt.Printf("%d ", it.Val)
	}
	fmt.Printf("\n")
}
*/

/* 两数只和
1设计一个缓存把出现的value和index记录，为了方便后续查找用map<vlaue,index>形式存储
2.和 - 遍历到的数做key去map查找是否已经便利过
*/

/* func mySum(nums []int, target int) []int {
	rtn := make([]int, 0)
	cache := make(map[int]int, 0)

	for k, v := range nums {
		if pos, ok := cache[target-v]; ok {
			rtn = append(rtn, pos)
			rtn = append(rtn, k)

			break
		} else {
			cache[v] = k
		}
	}

	return rtn
}

func main() {
	nums := []int{3, 0, 7, 9}

	fmt.Printf("%v", mySum(nums, 10))
} */
