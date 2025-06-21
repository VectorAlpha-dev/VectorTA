import React from 'react';
import { SearchButton } from '../search/SearchButton';
import { searchIndex } from '../../data/search-index';

export function Search() {
  return <SearchButton searchItems={searchIndex} />;
}