# redis

### redis分布式锁
&emsp;&emsp;crossRedsis公共的redis，各个跨服组都可以写，放一些排行榜等，公共数据。而一些数据，只能使用Set，zAdd等而不能使用hincrby等redis提供的自增原子操作，这样就会出现复写问题，为了解决这类并发问题，就要使用redis分布式锁。redis官网也可以直接搜到分布式锁知识。设置一个锁的代码如下：
```go
//用redis setnx 对key加锁
func (this *RedisDb) SetNxLock(lockKey string, lockTime int, now time.Time, isWaiting bool) (bool, error) {
retry:
	flag, err := this.SetNx(lockKey, now.Unix()+int64(lockTime)).Int()
	if err != nil {
		return false, errors.New(fmt.Sprintf("SetNxProcess error lockkey:%s err:%s", lockKey, err.Error()))
	}

	if flag == 1 { //加锁成功
		return true, nil
	} else { //加锁失败
		if expireTime, err := this.Get(lockKey).Int(); err == nil {
			if expireTime > int(now.Unix()) { //锁未过期
				if isWaiting {
					if time.Now().Unix()-now.Unix() > 5*60 {
						return false, errors.New("max timeout 5 minutes triggered")
					}
					time.Sleep(1 * time.Millisecond)
					goto retry
				} else {
					return false, errors.New("fail")
				}
			} else {
                //getset原子操作，这样做的原因是，一旦get返回的大于期望返回的值，那么说明已经有别的地方执行了set，
                //而本次set已经覆盖了别人set的值,这样就认为本次set的是失败，根据要求继续等待，或者返回失败
				expireTime, err = this.GetSet(lockKey, now.Unix()+int64(lockTime)).Int()
				if expireTime < int(now.Unix()) { //加锁成功
					return true, nil
				} else { //加锁失败，其他用户先加锁了
					if isWaiting {
						if time.Now().Unix()-now.Unix() > 5*60 {
							return false, errors.New("max timeout 5 minutes triggered")
						}
						time.Sleep(1 * time.Millisecond)
						goto retry
					} else {
						return false, errors.New("fail")
					}
				}
			}
		}
	}
	return false, errors.New("SetNxProcess lock error")
}
```
&emsp;&emsp;在完成对公共的redis写操作，删除掉lockKey,这样就让出给下一个想要写入的操作。
***
### redis 缓存穿透、缓存击穿、缓存雪崩

* **缓存穿透**：key对应的数据在数据源并不存在，每次针对此key的请求从缓存获取不到，请求都会到数据源，从而可能压跨数据源。比如用一个不存在的用户id获取用户的信息，不论缓存还是数据库都没有，若黑客利用此漏洞进行攻击可能压跨数据库。

* **缓存击穿**：key对应的数据存在，但是redis中过期，若大量的同时并发请求，这些请求发现缓存过期一般都会从后端DB加载数据并设置缓存，这个时候高并发的请求可能会瞬间把后端DB压跨。

* **缓存穿透**：当缓存服务器重启或者大量缓存集中在某一个时间段失效的时候，也会给后端系统（比如DB）带来很大的压力。

#### 解决思路

* **缓存穿透**：一个一定不存在缓存及查询不到的数据，由于缓存是不命中时被动写的，并且处于容错考虑。如果从存储层查不到数据则不写入缓存，这将导致这个不存在的数据每次请求都要到存储层去查询，失去了缓存的意义。有很多种方法可以有效地解决缓存穿透的问题。

1. 布隆过滤器：直观的说，bloom算法类似一个hash set，用来判断某个元素（key）是否某个集合中。和一般的hash set不同的是，这个算法无需存储key的值，对于每个key，只需要k个比特位，每个存储一个标志，用来判断是否在集合中。后端插入一个key时，在这个服务中设置一次，需要查询后端时，**先判断key在后端是否存在**，这样能避免后端的压力。
2. 另一种简单粗暴的方法，如果一个查询返回为空（不管是数据不存在，还是系统故障），仍然**把这个空结果进行缓存**，但设置一个比较的短的、合理的时间让它过期。

* **缓存击穿** key可能会在某些时间点被超高并发地访问（0点抢购），是一种非常“热点”的数据。这个时候。需要考虑一个问题：缓存被“击穿”的问题。简单的说，就是在缓存失效的时候（判断拿出来的值为空），不是立即去load DB（**去查看是否有别人在加载，没人加载数据再去加载，有人加载了，等待一会直接从缓存读去**），而是先使用缓存工具的某些带成功操作的返回值操作（比如Redis的SETNX或者Memecaceh的ADD）去set一个mutex key，当操作返回成功时，再进行load DB操作并回设缓存；否则，就重试整个get缓存的方法。（load的人用redis分布式锁加锁,有点类似单例的模式的懒汉模式的思路）

* **缓存雪崩** 这个问题是针对多个key，前面两个问题是一个key。多个key同时失效，而引起对数据库大量访问。大多数系统设计者考虑用加锁或者队列的方式保证不会有大量的线程对数据库一次性读写，从而避免失效时大量的并发请求落到底层存储系统上。还有一个简单方案就是将缓存失效时间分散开，比可以在原有的失效时间基础上增加一个随机值，这样每一个缓存的过期时间的重复率就会降低，就很难引发集体是失效的事件。
***
#### [参考链接](https://www.cnblogs.com/xichji/p/11286443.html)
