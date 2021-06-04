package main

import (
	"fmt"
	"time"
)

func main() {
	//t, _ := time.Parse("15:04:05", "21:19:00")
	// data := time.Now().Month()
	// now := time.Now().Month()
	// fmt.Println(t.Hour())
	// fmt.Println(t.Minute())
	// fmt.Println(t.Second())

	//fmt.Println(t)
	// fmt.Println(time.Now().UTC().Date())
	// fmt.Println(time.Now().Date())
	// fmt.Println(time.Now().Year())
	// fmt.Println(time.Now().YearDay())
	fmt.Println(time.Now().Month())
	fmt.Println(int(time.Now().Month()))
	// fmt.Println(time.Now().Day())
}

/*
func main() {
	props := make(map[int]int)

	//var test map[int]int

	fmt.Println(props[1])
	fmt.Println(props[1] == 2)
	fmt.Println(len(props))

	props[1] = 1

	fmt.Println(len(props))
}
*/
//
/*
func GetDateWithOffset(t time.Time) string {
	return t.Add(-time.Hour * 5).Format("060102")
}

func main() {
	fmt.Println(GetDateWithOffset(time.Now()))
	fmt.Println(time.Now().Add(-time.Hour * 18).Format("060102"))

}


func main() {
	t := 2
	h, _ := time.ParseDuration(strconv.Itoa(t) + "h")

	//t := 2
	now := time.Now()
	t1 := now.Add(h)

	fmt.Println(now)
	fmt.Println(t1)

}



func main() {
	n := int32(648)
	test := float32(2.2)

	cost := float32(n) * test

	fmt.Println(cost)
	fmt.Println(int32(cost))

	test2 := float32(2.2) / float32(0.1)
	fmt.Println(test2)

}


func main() {
	fmt.Println(time.Unix(1611386871, 0).Format("2006-01-02 15:04:05"))
	fmt.Println(time.Unix(1611386871, 0).Year())
	fmt.Println(time.Unix(1611386871, 0).Month())
	fmt.Println(time.Unix(1611386871, 0).Date())

	fmt.Println(time.Unix(1611386871, 0).YearDay())
}


func main() {
	println("%d\n", rand.Intn(2))
	println("%d\n", rand.Intn(2))
	println("%d\n", rand.Intn(2))
	println("%d\n", rand.Intn(2))
	println("%d\n", rand.Intn(2))
	println("%d\n", rand.Intn(2))
	println("%d\n", rand.Intn(2))
	println("%d\n", rand.Intn(2))
	println("%d\n", rand.Intn(2))
}


/////////////////////////////////////////////////
/////////////////////拓扑排序/////////////////////
/////////////////////////////////////////////////

// prereqs记录了每个课程的前置课程
var prereqs = map[string][]string{
	"algorithms": {"data structures"},
	"calculus":   {"linear algebra"},
	"compilers": {
		"data structures",
		"formal languages",
		"computer organization",
	},
	"data structures":       {"discrete math"},
	"databases":             {"data structures"},
	"discrete math":         {"intro to programming"},
	"formal languages":      {"discrete math"},
	"networks":              {"operating systems"},
	"operating systems":     {"data structures", "computer organization"},
	"programming languages": {"data structures", "computer organization"},
}

func main() {
	//for i, course := range myTopoSort(prereqs) {
	for i, course := range topoSort(prereqs) {
		fmt.Printf("%d:\t%s\n", i+1, course)
	}
}

func myTopoSort(m map[string][]string) []string {
	var order []string
	seen := make(map[string]bool)
	var visitAll func([]string)

	visitAll = func(items []string) {
		for _, item := range items {
			if !seen[item] {
				seen[item] = true

				visitAll(m[item])

				order = append(order, item)
			}
		}
	}

	var keys []string
	for key := range m {
		keys = append(keys, key)
	}

	//sort.Strings(keys)
	visitAll(keys)

	return order
}

func topoSort(m map[string][]string) []string {
	var order []string
	seen := make(map[string]bool)
	var visitAll func(items []string)

	visitAll = func(items []string) {
		for _, item := range items {
			if !seen[item] {
				seen[item] = true

				visitAll(m[item])

				order = append(order, item)
			}
		}
	}

	var keys []string
	for key := range m {
		keys = append(keys, key)
	}

	sort.Strings(keys)
	visitAll(keys)

	return order
}


// squares返回一个匿名函数。
// 该匿名函数每次被调用时都会返回下一个数的平方。
func(x) int {
	x++
	return x * x
}

func squares() func() int {
	var x int
	return func()
	}
}
func main() {
	f := squares()
	fmt.Println(f()) // "1"
	fmt.Println(f()) // "4"
	fmt.Println(f()) // "9"
	fmt.Println(f()) // "16"
}


func main() {
	ch := make(chan string)

	go sendData(ch)
	go getData(ch)

	time.Sleep(1e9)
}

func sendData(ch chan string) {
	ch <- "Washington"
	ch <- "Tripoli"
	ch <- "London"
	ch <- "Beijing"
	ch <- "Tokio"
}

func getData(ch chan string) {
	var input string
	// time.Sleep(1e9)
	for {
		input = <-ch
		fmt.Printf("%s ", input)
	}
}

func main() {
	// Version A:
	items := make([]map[int]int, 5)
	for i := range items {
		items[i] = make(map[int]int, 1)
		items[i][1] = 2
	}

	fmt.Printf("Version A: Value of items: %v\n", items)

	// Version B: NOT GOOD!
	items2 := make([]map[int]int, 5)
	for _, item := range items2 {
		item = make(map[int]int, 1) // item is only a copy of the slice element.
		item[1] = 2                 // This 'item' will be lost on the next iteration.
	}

	fmt.Printf("Version B: Value of items: %v\n", items2)
}

import (
	"fmt"
)

func main() {
	tag := true

	if true {
		tag = false
	}

	fmt.Printf("time:%v", tag)
}

func main() {
	slice1 := make([]int, 0, 10)
	// load the slice, cap(slice1) is 10:
	for i := 0; i < cap(slice1); i++ {
		slice1 = slice1[0 : i+1]
		slice1[i] = i
		fmt.Printf("The length of slice is %d\n", len(slice1))
	}

	// print the slice:
	for i := 0; i < len(slice1); i++ {
		fmt.Printf("Slice at %d is %d\n", i, slice1[i])
	}

	fmt.Printf("Slice len test %d\n", len(slice1[5:5]))
	fmt.Printf("Slice len test %d\n", len(slice1[5:6]))

}

package main

import "fmt"

func main() {
	// make an Add2 function, give it a name p2, and call it:
	p2 := Add2()
	fmt.Printf("Call Add2 for 3 gives: %v\n", p2(3))
	// make a special Adder function, a gets value 2:
	TwoAdder := Adder(2)
	fmt.Printf("The result is: %v\n", TwoAdder(3))
}

func Add2() func(b int) int {
	return func(b int) int {
		return b + 2
	}
}

func Adder(a int) func(b int) int {
	return func(b int) int {
		return a + b
	}
}

package main

import (
	"fmt"
	"io"
	"log"
)

func func1(s string) (n int, err error) {
	defer func() {
		fmt.Printf("hello\n")
		log.Printf("hello1")
		log.Printf("func1(%q) = %d, %v", s, n, err)
	}()
	return 7, io.EOF
}

func main() {
	func1("Go")
}

package main

import "fmt"

func main() {
	x := min(1, 3, 2, 0)
	fmt.Printf("The minimum is: %d\n", x)
	slice := []int{7, 9, 3, 5, 1}
	x = min(slice...)
	fmt.Printf("The minimum in the slice is: %d", x)
}

func min(s ...int) int {
	if len(s) == 0 {
		return 0
	}
	min := s[0]
	for _, v := range s {
		if v < min {
			min = v
		}
	}
	return min
}

package main

import "fmt"

func f(a [3]int)   { fmt.Println(a) }
func fp(a *[3]int) { fmt.Println(a) }

func main() {
	var ar [3]int
	f(ar)   // passes a copy of ar
	fp(&ar) // passes a pointer to ar
}

package main

import "fmt"

func main() {
	a := [...]string{"a", "b", "c", "d"}
	for i := range a {
		fmt.Println("Array item", i, "is", a[i])
	}
}

package main

var a string

func main() {
	b := "hello world1\n"
	a = `hello world2\n`
	c := "hello world3\n"
	print(a)
	print(b)
	print(c)
	//f1()cle
}


func f1() {
	a := "O"
	print(a)
	f2()
}

func f2() {
	print(a)
}*/
