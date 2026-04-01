from fastapi import APIRouter, HTTPException, status

from backend.schemas.agents import AgentCreate, AgentListResponse, AgentResponse, AgentUpdate
from backend.services.ingestion_factory import get_agent_store, set_active_models


router = APIRouter(prefix="/agents", tags=["agents"])


def _agent_to_response(agent) -> AgentResponse:
    from backend.schemas.agents import AgentConfigSchema

    return AgentResponse(
        agent_id=agent.agent_id,
        name=agent.name,
        description=agent.description,
        config=AgentConfigSchema(
            system_prompt=agent.config.system_prompt,
            dataset_id=agent.config.dataset_id,
            embedding_model=agent.config.embedding_model,
            chat_model=agent.config.chat_model,
            temperature=agent.config.temperature,
            top_k=agent.config.top_k,
        ),
        created_at=agent.created_at,
        updated_at=agent.updated_at,
    )


@router.post("/", response_model=AgentResponse, status_code=status.HTTP_201_CREATED)
async def create_agent(payload: AgentCreate) -> AgentResponse:
    from rag.storage.agent_store import AgentConfig

    store = get_agent_store()
    config = AgentConfig(
        system_prompt=payload.config.system_prompt,
        dataset_id=payload.config.dataset_id,
        embedding_model=payload.config.embedding_model,
        chat_model=payload.config.chat_model,
        temperature=payload.config.temperature,
        top_k=payload.config.top_k,
    )

    agent = store.create(
        name=payload.name,
        description=payload.description,
        config=config,
    )

    return _agent_to_response(agent)


@router.get("/", response_model=AgentListResponse)
async def list_agents() -> AgentListResponse:
    store = get_agent_store()
    agents = store.list()
    return AgentListResponse(
        agents=[_agent_to_response(a) for a in agents],
        total=len(agents),
    )


@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(agent_id: str) -> AgentResponse:
    store = get_agent_store()
    agent = store.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    return _agent_to_response(agent)


@router.put("/{agent_id}", response_model=AgentResponse)
async def update_agent(agent_id: str, payload: AgentUpdate) -> AgentResponse:
    from rag.storage.agent_store import AgentConfig

    store = get_agent_store()
    updates: dict = {}
    if payload.name is not None:
        updates["name"] = payload.name
    if payload.description is not None:
        updates["description"] = payload.description
    if payload.config is not None:
        updates["config"] = {
            "system_prompt": payload.config.system_prompt,
            "dataset_id": payload.config.dataset_id,
            "embedding_model": payload.config.embedding_model,
            "chat_model": payload.config.chat_model,
            "temperature": payload.config.temperature,
            "top_k": payload.config.top_k,
        }

    agent = store.update(agent_id, updates)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    return _agent_to_response(agent)


@router.delete("/{agent_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_agent(agent_id: str) -> None:
    store = get_agent_store()
    deleted = store.delete(agent_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
