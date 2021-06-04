# hoodinn

### fight server

1. 打副本，只和自己的数据有关
2. 静态战斗服，例如押镖。多个服务器的人可以在一个战斗场景一起战斗，并非合服。
3. 

#### 个人副本
1. HandleCopyReadyReq
```go
type CopyReadyReq struct {
	CopyId  int32       //根据这个id回去，copy.xlsx配置中拉去配置信息
	ObjType int32       //0
	Line    int32       //0
}
```
2. 
> `HandleCopyReadyReq`
>> `func (this *FightManager) EnterFight`
>>> `factory.EnterChecker(user, copyId, line, nil) //在redis中查找fightId`
>>> `m.Fight.GetFightActorNoHero //m，为ModuleManager.收集任务的各个信息（战将，神兽等）创建战斗用户`

>>> `func (this *FSManager) RpcCal // // 本服战斗服+静态战斗跨服 RpcCal`

```go
//给gateserver推送消息
gateSeq := conn.GetSession().((*managers.ClientSession)).GateSeq

sessionId := conn.GetSession().((*managers.ClientSession)).GetId()
```


