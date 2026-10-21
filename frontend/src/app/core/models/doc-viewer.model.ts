export interface DocTab {
  id: string; // `${repoId || 'default'}:${filepath}`
  filepath: string;
  filename: string;
  repoId?: string;
  startLine?: number;
  endLine?: number;
  fullContent?: string;
  totalLines?: number;
  loading: boolean;
  error?: string;
  viewMode: 'rendered' | 'raw'; // для Markdown: превью или код
  scopeMode: 'full' | 'chunk'; // полный файл или только фрагмент
}
