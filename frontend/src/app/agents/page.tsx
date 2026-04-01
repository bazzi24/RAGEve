"use client";

import { useCallback, useEffect, useState } from "react";
import { useAgentsStore } from "@/stores/useAgentsStore";
import { useDatasetsStore } from "@/stores/useDatasetsStore";
import { useModelStore } from "@/stores/useModelStore";
import { useToastStore } from "@/stores/useToastStore";
import {
  listAgents,
  createAgent,
  updateAgent,
  deleteAgent,
} from "@/lib/api/agents";
import { listDatasets } from "@/lib/api/datasets";
import type { AgentResponse } from "@/lib/types";
import { Button } from "@/components/ui/Button";
import { Input, Textarea } from "@/components/ui/Input";
import { Select } from "@/components/ui/Select";
import { Slider } from "@/components/ui/Slider";
import { Modal } from "@/components/ui/Modal";
import styles from "./AgentsPage.module.css";

export default function AgentsPage() {
  const { agents, setAgents, addAgent, updateAgent: updateAgentInStore, removeAgent } = useAgentsStore();
  const { datasets, setDatasets } = useDatasetsStore();
  const { availableModels, embeddingModel: defaultEmbed, chatModel: defaultChat } = useModelStore();
  const { addToast } = useToastStore();

  const [showModal, setShowModal] = useState(false);
  const [editingAgent, setEditingAgent] = useState<AgentResponse | null>(null);
  const [saving, setSaving] = useState(false);

  // Form state
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [systemPrompt, setSystemPrompt] = useState("You are a helpful AI assistant.");
  const [datasetId, setDatasetId] = useState("");
  const [embedModel, setEmbedModel] = useState("");
  const [chatModel, setChatModel] = useState("");
  const [temperature, setTemperature] = useState(0.7);
  const [topK, setTopK] = useState(5);

  useEffect(() => {
    listAgents()
      .then((r) => setAgents(r.agents))
      .catch((e) => addToast(`Failed to load agents: ${e.message}`, "error"));
  }, [setAgents, addToast]);

  // Fetch all datasets from Qdrant so the dataset selector always shows what's available
  useEffect(() => {
    listDatasets()
      .then((res) => setDatasets(res.datasets))
      .catch((e) => addToast(`Failed to load datasets: ${e.message}`, "error"));
  }, [setDatasets, addToast]);

  const openCreate = () => {
    setEditingAgent(null);
    setName("");
    setDescription("");
    setSystemPrompt("You are a helpful AI assistant.");
    setDatasetId(datasets[0]?.dataset_id || "");
    setEmbedModel(defaultEmbed || availableModels[0] || "");
    setChatModel(defaultChat || availableModels[0] || "");
    setTemperature(0.7);
    setTopK(5);
    setShowModal(true);
  };

  const openEdit = (agent: AgentResponse) => {
    setEditingAgent(agent);
    setName(agent.name);
    setDescription(agent.description);
    setSystemPrompt(agent.config.system_prompt);
    setDatasetId(agent.config.dataset_id);
    setEmbedModel(agent.config.embedding_model);
    setChatModel(agent.config.chat_model);
    setTemperature(agent.config.temperature);
    setTopK(agent.config.top_k);
    setShowModal(true);
  };

  const handleSave = async () => {
    if (!name.trim()) {
      addToast("Agent name is required", "warning");
      return;
    }
    setSaving(true);
    try {
      const payload = {
        name: name.trim(),
        description: description.trim(),
        config: {
          system_prompt: systemPrompt,
          dataset_id: datasetId,
          embedding_model: embedModel,
          chat_model: chatModel,
          temperature,
          top_k: topK,
        },
      };
      if (editingAgent) {
        const updated = await updateAgent(editingAgent.agent_id, payload);
        updateAgentInStore(updated);
        addToast(`Agent "${updated.name}" updated`, "success");
      } else {
        const created = await createAgent(payload);
        addAgent(created);
        addToast(`Agent "${created.name}" created`, "success");
      }
      setShowModal(false);
    } catch (err) {
      addToast(`Save failed: ${err instanceof Error ? err.message : String(err)}`, "error");
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async (agent: AgentResponse) => {
    try {
      await deleteAgent(agent.agent_id);
      removeAgent(agent.agent_id);
      addToast(`Agent "${agent.name}" deleted`, "info");
    } catch {
      addToast(`Delete failed`, "error");
    }
  };

  const modelOptions = availableModels.map((m) => ({ value: m, label: m }));
  const datasetOptions = datasets.map((d) => ({ value: d.dataset_id, label: d.dataset_id }));

  return (
    <div className={styles.page}>
      <div className={styles.header}>
        <h1 className={styles.title}>Agents</h1>
        <Button onClick={openCreate}>New Agent</Button>
      </div>

      {agents.length === 0 ? (
        <div className={styles.emptyState}>
          No agents yet. Create one to start chatting.
        </div>
      ) : (
        <div className={styles.list}>
          {agents.map((agent) => (
            <div key={agent.agent_id} className={styles.card}>
              <div className={styles.cardTop}>
                <div>
                  <div className={styles.cardTitle}>{agent.name}</div>
                  {agent.description && (
                    <div className={styles.cardDesc}>{agent.description}</div>
                  )}
                </div>
                <div className={styles.cardActions}>
                  <Button variant="ghost" size="sm" onClick={() => openEdit(agent)}>Edit</Button>
                  <Button variant="danger" size="sm" onClick={() => handleDelete(agent)}>Delete</Button>
                </div>
              </div>
              <div className={styles.cardBody}>
                <div className={styles.field}>
                  <span className={styles.fieldLabel}>Dataset</span>
                  <span className={styles.fieldValue}>{agent.config.dataset_id}</span>
                </div>
                <div className={styles.field}>
                  <span className={styles.fieldLabel}>Embed Model</span>
                  <span className={styles.fieldValue}>{agent.config.embedding_model}</span>
                </div>
                <div className={styles.field}>
                  <span className={styles.fieldLabel}>Chat Model</span>
                  <span className={styles.fieldValue}>{agent.config.chat_model}</span>
                </div>
                <div className={styles.field}>
                  <span className={styles.fieldLabel}>Temperature</span>
                  <span className={styles.fieldValue}>{agent.config.temperature}</span>
                </div>
                <div className={styles.field}>
                  <span className={styles.fieldLabel}>Top-K</span>
                  <span className={styles.fieldValue}>{agent.config.top_k}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      <Modal
        open={showModal}
        onClose={() => setShowModal(false)}
        title={editingAgent ? "Edit Agent" : "New Agent"}
        footer={
          <>
            <Button variant="ghost" onClick={() => setShowModal(false)}>Cancel</Button>
            <Button onClick={handleSave} loading={saving}>{editingAgent ? "Save Changes" : "Create"}</Button>
          </>
        }
      >
        <div className={styles.form}>
          <div className={styles.formGrid}>
            <Input label="Name" value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. Research Assistant" />
            <Input label="Description" value={description} onChange={(e) => setDescription(e.target.value)} placeholder="Optional" />
          </div>

          <div className={styles.formFull}>
            <Textarea
              label="System Prompt"
              value={systemPrompt}
              onChange={(e) => setSystemPrompt(e.target.value)}
              rows={5}
            />
          </div>

          <div className={styles.formGrid}>
            <Select
              id="agent-dataset"
              label="Dataset"
              options={datasetOptions}
              value={datasetId}
              onChange={(e) => setDatasetId(e.target.value)}
              placeholder="Select dataset..."
            />
            <Select
              id="agent-embed"
              label="Embedding Model"
              options={modelOptions}
              value={embedModel}
              onChange={(e) => setEmbedModel(e.target.value)}
              placeholder="Select..."
            />
          </div>

          <div className={styles.formGrid}>
            <Select
              id="agent-chat"
              label="Chat Model"
              options={modelOptions}
              value={chatModel}
              onChange={(e) => setChatModel(e.target.value)}
              placeholder="Select..."
            />
            <Slider
              id="agent-temp"
              label="Temperature"
              min={0}
              max={2}
              step={0.1}
              value={temperature}
              onChange={(e) => setTemperature(Number(e.target.value))}
              valueFormatter={(v) => v.toFixed(1)}
            />
          </div>

          <Slider
            id="agent-topk"
            label="Top-K Retrieval"
            min={1}
            max={20}
            step={1}
            value={topK}
            onChange={(e) => setTopK(Number(e.target.value))}
            valueFormatter={(v) => String(v)}
          />
        </div>
      </Modal>
    </div>
  );
}
