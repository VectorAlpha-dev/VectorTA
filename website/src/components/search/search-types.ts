export interface SearchItem {
  id: string;
  title: string;
  description: string;
  category: string;
  url: string;
  type: 'indicator' | 'guide' | 'page';
}