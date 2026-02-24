import asyncio
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class RequestVote:
    term: int
    candidate_id: int
    last_log_index: int
    last_log_term: int


@dataclass
class RequestVoteResponse:
    term: int
    vote_granted: bool
    voter_id: int


@dataclass
class AppendEntries:
    term: int
    leader_id: int
    prev_log_index: int
    prev_log_term: int
    entries: List[Tuple[int, Any]]  # (term, command)
    leader_commit: int


@dataclass
class AppendEntriesResponse:
    term: int
    success: bool
    follower_id: int
    match_index: int


class Network:
    """
    Asynchronous message transport with bounded random delay.
    Messages are delayed independently; loss and duplication are not modeled.
    """
    def __init__(self, delay_range: Tuple[float, float] = (0.002, 0.02)):
        self.delay_range = delay_range
        self.nodes: Dict[int, "RaftNode"] = {}

    def register(self, node: "RaftNode") -> None:
        self.nodes[node.node_id] = node

    async def send(self, to_id: int, msg: Any) -> None:
        await asyncio.sleep(random.uniform(*self.delay_range))
        await self.nodes[to_id].inbox.put(msg)


class RaftNode:
    """
    Minimal Raft-style replicated state machine.

    Safety scope of this implementation:
    - Crash faults are not simulated; persistence is not implemented.
    - Network does not drop or duplicate messages.
    - Membership is fixed; no reconfiguration is supported.
    - Log matching and leader election follow the Raft structure.
    """
    def __init__(
        self,
        node_id: int,
        peer_ids: List[int],
        net: Network,
        election_timeout_range: Tuple[float, float] = (0.08, 0.16),
        heartbeat_interval: float = 0.03,
    ):
        self.node_id = node_id
        self.peer_ids = peer_ids
        self.net = net

        self.inbox: asyncio.Queue[Any] = asyncio.Queue()

        # Persistent state (in-memory for this reference implementation)
        self.current_term: int = 0
        self.voted_for: Optional[int] = None
        self.log: List[Tuple[int, Any]] = []  # (term, command)

        # Volatile state
        self.commit_index: int = -1
        self.last_applied: int = -1

        # Replicated state machine
        self.counter: int = 0

        # Role and leader tracking
        self.role: str = "follower"  # follower | candidate | leader
        self.leader_id: Optional[int] = None

        # Leader-only state
        self.next_index: Dict[int, int] = {}
        self.match_index: Dict[int, int] = {}

        self.election_timeout_range = election_timeout_range
        self.heartbeat_interval = heartbeat_interval
        self._votes_received: set[int] = set()
        self._running = True
        self._task: Optional[asyncio.Task] = None

        self._reset_election_deadline()

    def _reset_election_deadline(self) -> None:
        loop_time = asyncio.get_event_loop().time()
        self.election_deadline = loop_time + random.uniform(*self.election_timeout_range)

    def last_log_index(self) -> int:
        return len(self.log) - 1

    def last_log_term(self) -> int:
        return self.log[-1][0] if self.log else 0

    def _log_up_to_date(self, cand_index: int, cand_term: int) -> bool:
        my_term = self.last_log_term()
        if cand_term != my_term:
            return cand_term > my_term
        return cand_index >= self.last_log_index()

    def _become_follower(self, term: int, leader_id: Optional[int] = None) -> None:
        prev_term = self.current_term
        self.role = "follower"
        self.current_term = term
        self.leader_id = leader_id
        if term > prev_term:
            self.voted_for = None
        self._votes_received.clear()
        self.next_index.clear()
        self.match_index.clear()
        self._reset_election_deadline()

    def _become_leader(self) -> None:
        self.role = "leader"
        self.leader_id = self.node_id
        next_i = len(self.log)
        self.next_index = {pid: next_i for pid in self.peer_ids}
        self.match_index = {pid: -1 for pid in self.peer_ids}
        self.election_deadline = asyncio.get_event_loop().time() + self.heartbeat_interval

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run(self) -> None:
        while self._running:
            now = asyncio.get_event_loop().time()
            tick = max(0.0, min(0.05, self.election_deadline - now))

            try:
                msg = await asyncio.wait_for(self.inbox.get(), timeout=tick)
                await self._handle(msg)
            except asyncio.TimeoutError:
                if self.role != "leader" and asyncio.get_event_loop().time() >= self.election_deadline:
                    await self._start_election()
                if self.role == "leader":
                    await self._send_heartbeats()

            await self._apply_committed()

    async def _handle(self, msg: Any) -> None:
        if isinstance(msg, RequestVote):
            await self._on_request_vote(msg)
        elif isinstance(msg, RequestVoteResponse):
            await self._on_request_vote_response(msg)
        elif isinstance(msg, AppendEntries):
            await self._on_append_entries(msg)
        elif isinstance(msg, AppendEntriesResponse):
            await self._on_append_entries_response(msg)
        elif isinstance(msg, tuple) and msg and msg[0] == "client_command":
            _, cmd = msg
            await self._on_client_command(cmd)

    async def _start_election(self) -> None:
        self.role = "candidate"
        self.current_term += 1
        self.voted_for = self.node_id
        self.leader_id = None
        self._votes_received = {self.node_id}
        self._reset_election_deadline()

        rv = RequestVote(
            term=self.current_term,
            candidate_id=self.node_id,
            last_log_index=self.last_log_index(),
            last_log_term=self.last_log_term(),
        )
        await asyncio.gather(*(self.net.send(pid, rv) for pid in self.peer_ids))

    async def _on_request_vote(self, m: RequestVote) -> None:
        if m.term < self.current_term:
            await self.net.send(m.candidate_id, RequestVoteResponse(self.current_term, False, self.node_id))
            return

        if m.term > self.current_term:
            self._become_follower(m.term)

        vote_granted = False
        if (self.voted_for is None or self.voted_for == m.candidate_id) and self._log_up_to_date(m.last_log_index, m.last_log_term):
            self.voted_for = m.candidate_id
            vote_granted = True
            self._reset_election_deadline()

        await self.net.send(m.candidate_id, RequestVoteResponse(self.current_term, vote_granted, self.node_id))

    async def _on_request_vote_response(self, m: RequestVoteResponse) -> None:
        if self.role != "candidate":
            return

        if m.term > self.current_term:
            self._become_follower(m.term)
            return

        if m.term < self.current_term:
            return

        if m.vote_granted:
            self._votes_received.add(m.voter_id)
            if len(self._votes_received) > (len(self.peer_ids) + 1) // 2:
                self._become_leader()

    async def _send_heartbeats(self) -> None:
        await asyncio.gather(*(self._send_append_entries(pid) for pid in self.peer_ids))
        self.election_deadline = asyncio.get_event_loop().time() + self.heartbeat_interval

    async def _send_append_entries(self, pid: int) -> None:
        next_i = self.next_index.get(pid, len(self.log))
        next_i = max(0, min(next_i, len(self.log)))
        self.next_index[pid] = next_i

        prev_i = next_i - 1
        prev_term = self.log[prev_i][0] if 0 <= prev_i < len(self.log) else 0
        entries = self.log[next_i:]

        ae = AppendEntries(
            term=self.current_term,
            leader_id=self.node_id,
            prev_log_index=prev_i,
            prev_log_term=prev_term,
            entries=entries,
            leader_commit=self.commit_index,
        )
        await self.net.send(pid, ae)

    async def _on_append_entries(self, m: AppendEntries) -> None:
        if m.term < self.current_term:
            await self.net.send(m.leader_id, AppendEntriesResponse(self.current_term, False, self.node_id, -1))
            return

        if m.term > self.current_term or self.role != "follower":
            self._become_follower(m.term, leader_id=m.leader_id)
        else:
            self.leader_id = m.leader_id
            self._reset_election_deadline()

        if m.prev_log_index >= 0:
            if m.prev_log_index >= len(self.log):
                await self.net.send(m.leader_id, AppendEntriesResponse(self.current_term, False, self.node_id, -1))
                return
            if self.log[m.prev_log_index][0] != m.prev_log_term:
                await self.net.send(m.leader_id, AppendEntriesResponse(self.current_term, False, self.node_id, -1))
                return

        insert_at = m.prev_log_index + 1
        appended = 0

        i = 0
        while i < len(m.entries) and (insert_at + i) < len(self.log):
            idx = insert_at + i
            if self.log[idx] != m.entries[i]:
                self.log = self.log[:idx]
                break
            i += 1

        for j in range(i, len(m.entries)):
            self.log.append(m.entries[j])
            appended += 1

        if m.leader_commit > self.commit_index:
            self.commit_index = min(m.leader_commit, len(self.log) - 1)

        if appended > 0:
            match_index = insert_at + appended - 1
        else:
            match_index = m.prev_log_index

        match_index = min(match_index, len(self.log) - 1)
        await self.net.send(m.leader_id, AppendEntriesResponse(self.current_term, True, self.node_id, match_index))

    async def _on_append_entries_response(self, m: AppendEntriesResponse) -> None:
        if self.role != "leader":
            return

        if m.term > self.current_term:
            self._become_follower(m.term)
            return

        if not m.success:
            self.next_index[m.follower_id] = max(0, self.next_index.get(m.follower_id, len(self.log)) - 1)
            await self._send_append_entries(m.follower_id)
            return

        capped_match = min(m.match_index, len(self.log) - 1)
        self.match_index[m.follower_id] = capped_match
        self.next_index[m.follower_id] = capped_match + 1

        match = list(self.match_index.values()) + [len(self.log) - 1]
        match.sort()
        majority_match = match[len(match) // 2]

        if majority_match > self.commit_index and majority_match >= 0:
            if self.log[majority_match][0] == self.current_term:
                self.commit_index = majority_match

    async def _apply_committed(self) -> None:
        while self.last_applied < self.commit_index:
            self.last_applied += 1
            _, cmd = self.log[self.last_applied]
            if cmd == "inc":
                self.counter += 1

    async def _on_client_command(self, cmd: Any) -> None:
        if self.role != "leader":
            if self.leader_id is not None:
                await self.net.send(self.leader_id, ("client_command", cmd))
            return

        self.log.append((self.current_term, cmd))
        await asyncio.gather(*(self._send_append_entries(pid) for pid in self.peer_ids))


async def run_cluster() -> None:
    random.seed(3)
    net = Network()
    n = 5
    nodes: List[RaftNode] = []

    for i in range(n):
        peers = [j for j in range(n) if j != i]
        node = RaftNode(i, peers, net)
        net.register(node)
        nodes.append(node)

    for node in nodes:
        await node.start()

    await asyncio.sleep(0.6)

    for _ in range(25):
        await nodes[random.randrange(n)].inbox.put(("client_command", "inc"))
        await asyncio.sleep(0.005)

    await asyncio.sleep(1.0)

    for node in nodes:
        print(node.node_id, node.role, node.current_term, node.counter, node.commit_index, len(node.log), node.leader_id)

    await asyncio.gather(*(node.stop() for node in nodes))


if __name__ == "__main__":
    asyncio.run(run_cluster())
