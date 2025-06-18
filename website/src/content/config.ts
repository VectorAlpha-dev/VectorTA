import { defineCollection, z } from 'astro:content';

// Define the schema for indicator documentation
const indicatorsCollection = defineCollection({
  type: 'content',
  schema: z.object({
    title: z.string(),
    description: z.string(),
    category: z.enum([
      'trend',
      'momentum', 
      'volatility',
      'volume',
      'moving_averages',
      'oscillators',
      'other'
    ]),
    parameters: z.array(z.object({
      name: z.string(),
      type: z.enum(['number', 'boolean', 'string']),
      default: z.union([z.number(), z.boolean(), z.string()]),
      min: z.number().optional(),
      max: z.number().optional(),
      description: z.string()
    })),
    returns: z.object({
      type: z.string(),
      description: z.string()
    }),
    complexity: z.string().default('O(n)'),
    since: z.string().optional(),
    seeAlso: z.array(z.string()).optional(),
    references: z.array(z.object({
      title: z.string(),
      url: z.string().url()
    })).optional(),
    isDraft: z.boolean().default(false),
    implementationStatus: z.enum(['complete', 'in-progress', 'planned']).default('planned')
  })
});

// Define the schema for tutorial/guide content
const guidesCollection = defineCollection({
  type: 'content',
  schema: z.object({
    title: z.string(),
    description: z.string(),
    author: z.string().default('VectorTA Team'),
    publishDate: z.date(),
    updatedDate: z.date().optional(),
    tags: z.array(z.string()),
    difficulty: z.enum(['beginner', 'intermediate', 'advanced']),
    readingTime: z.number(), // in minutes
    isDraft: z.boolean().default(false)
  })
});

export const collections = {
  'indicators': indicatorsCollection,
  'guides': guidesCollection
};