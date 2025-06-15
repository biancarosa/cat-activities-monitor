"use client";

import { useState, useEffect, useCallback } from 'react';
import { Plus, Edit, Trash2, Cat, Clock, Target } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle, AlertDialogTrigger } from '@/components/ui/alert-dialog';
import { catProfileApi, type CatProfile, type CreateCatProfileRequest, type UpdateCatProfileRequest } from '@/lib/api';
import { getCatColor } from '@/lib/colors';

export default function CatProfilesPage() {
  const [profiles, setProfiles] = useState<CatProfile[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false);
  const [editingProfile, setEditingProfile] = useState<CatProfile | null>(null);

  const loadProfiles = useCallback(async () => {
    try {
      setLoading(true);
      const response = await catProfileApi.list();
      console.log('API Response:', response); // Debug logging
      setProfiles(Array.isArray(response.cats) ? response.cats : []);
      setError(null);
    } catch (err) {
      console.error('Failed to load cat profiles:', err);
      setError('Failed to load cat profiles. Please check your API connection.');
      setProfiles([]); // Set empty array on error
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadProfiles();
  }, [loadProfiles]);

  const handleCreateProfile = async (profileData: CreateCatProfileRequest) => {
    try {
      await catProfileApi.create(profileData);
      await loadProfiles();
      setIsCreateDialogOpen(false);
    } catch (err) {
      console.error('Failed to create cat profile:', err);
      throw new Error('Failed to create cat profile');
    }
  };

  const handleUpdateProfile = async (catUuid: string, profileData: UpdateCatProfileRequest) => {
    try {
      await catProfileApi.update(catUuid, profileData);
      await loadProfiles();
      setEditingProfile(null);
    } catch (err) {
      console.error('Failed to update cat profile:', err);
      throw new Error('Failed to update cat profile');
    }
  };

  const handleDeleteProfile = async (catUuid: string) => {
    try {
      await catProfileApi.delete(catUuid);
      await loadProfiles();
    } catch (err) {
      console.error('Failed to delete cat profile:', err);
      alert('Failed to delete cat profile');
    }
  };

  if (loading) {
    return (
      <div className="container mx-auto p-6">
        <div className="flex items-center justify-center h-64">
          <div className="text-lg">Loading cat profiles...</div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="container mx-auto p-6">
        <div className="flex items-center justify-center h-64">
          <div className="text-lg text-red-500">{error}</div>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto p-6">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold">Cat Profiles</h1>
          <p className="text-muted-foreground mt-2">
            Manage your cat profiles and their unique identifiers
          </p>
        </div>
        
        <Dialog open={isCreateDialogOpen} onOpenChange={setIsCreateDialogOpen}>
          <DialogTrigger asChild>
            <Button>
              <Plus className="w-4 h-4 mr-2" />
              Add Cat Profile
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Create New Cat Profile</DialogTitle>
              <DialogDescription>
                Add a new cat to your monitoring system with a unique identifier.
              </DialogDescription>
            </DialogHeader>
            <CatProfileForm<CreateCatProfileRequest> onSubmit={handleCreateProfile} />
          </DialogContent>
        </Dialog>
      </div>

      {profiles.length === 0 ? (
        <Card>
          <CardContent className="flex flex-col items-center justify-center h-64">
            <Cat className="w-16 h-16 text-muted-foreground mb-4" />
            <h3 className="text-xl font-semibold mb-2">No Cat Profiles Yet</h3>
            <p className="text-muted-foreground text-center mb-4">
              Create your first cat profile to start tracking and identifying cats in your system.
            </p>
            <Button onClick={() => setIsCreateDialogOpen(true)}>
              <Plus className="w-4 h-4 mr-2" />
              Add First Cat Profile
            </Button>
          </CardContent>
        </Card>
      ) : (
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
          {Array.isArray(profiles) && profiles.map((profile) => (
            <CatProfileCard
              key={profile.cat_uuid}
              profile={profile}
              onEdit={setEditingProfile}
              onDelete={handleDeleteProfile}
            />
          ))}
        </div>
      )}

      {editingProfile && (
        <Dialog open={!!editingProfile} onOpenChange={() => setEditingProfile(null)}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Edit Cat Profile</DialogTitle>
              <DialogDescription>
                Update information for {editingProfile.name}.
              </DialogDescription>
            </DialogHeader>
            <CatProfileForm<UpdateCatProfileRequest>
              initialData={editingProfile}
              onSubmit={(data) => handleUpdateProfile(editingProfile.cat_uuid, data)}
            />
          </DialogContent>
        </Dialog>
      )}
    </div>
  );
}

interface CatProfileCardProps {
  profile: CatProfile;
  onEdit: (profile: CatProfile) => void;
  onDelete: (catUuid: string) => void;
}

function CatProfileCard({ profile, onEdit, onDelete }: CatProfileCardProps) {
  const catColor = getCatColor(profile.name, profile.cat_uuid);
  
  return (
    <Card className="group relative">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div 
              className="w-8 h-8 rounded-full flex items-center justify-center text-white font-semibold text-sm"
              style={{ backgroundColor: catColor }}
            >
              {profile.name.charAt(0).toUpperCase()}
            </div>
            <div>
              <CardTitle className="text-lg">{profile.name}</CardTitle>
              <CardDescription className="text-xs font-mono">
                {profile.cat_uuid.slice(0, 8)}...
              </CardDescription>
            </div>
          </div>
          <div className="flex gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
            <Button
              variant="ghost"
              size="icon"
              onClick={() => onEdit(profile)}
              className="h-8 w-8"
            >
              <Edit className="w-4 h-4" />
            </Button>
            <AlertDialog>
              <AlertDialogTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8 text-red-500 hover:text-red-600"
                >
                  <Trash2 className="w-4 h-4" />
                </Button>
              </AlertDialogTrigger>
              <AlertDialogContent>
                <AlertDialogHeader>
                  <AlertDialogTitle>Delete Cat Profile</AlertDialogTitle>
                  <AlertDialogDescription>
                    Are you sure you want to delete &ldquo;{profile.name}&rdquo;? This action cannot be undone.
                  </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <AlertDialogCancel>Cancel</AlertDialogCancel>
                  <AlertDialogAction 
                    onClick={() => onDelete(profile.cat_uuid)}
                    className="bg-red-600 hover:bg-red-700"
                  >
                    Delete
                  </AlertDialogAction>
                </AlertDialogFooter>
              </AlertDialogContent>
            </AlertDialog>
          </div>
        </div>
      </CardHeader>
      
      <CardContent className="space-y-4">
        {profile.description && (
          <p className="text-sm text-muted-foreground">{profile.description}</p>
        )}
        
        <div className="flex flex-wrap gap-2">
          {profile.color && (
            <Badge variant="outline" className="text-xs">
              {profile.color}
            </Badge>
          )}
          {profile.breed && (
            <Badge variant="outline" className="text-xs">
              {profile.breed}
            </Badge>
          )}
        </div>

        {profile.favorite_activities && profile.favorite_activities.length > 0 && (
          <div>
            <p className="text-sm font-medium mb-2">Favorite Activities:</p>
            <div className="flex flex-wrap gap-1">
              {profile.favorite_activities.map((activity, index) => (
                <Badge key={index} variant="secondary" className="text-xs">
                  {activity}
                </Badge>
              ))}
            </div>
          </div>
        )}

        <Separator />
        
        <div className="flex items-center justify-between text-sm text-muted-foreground">
          <div className="flex items-center gap-1">
            <Target className="w-4 h-4" />
            <span>{profile.total_detections} detections</span>
          </div>
          {profile.average_confidence > 0 && (
            <div>
              <span>{(profile.average_confidence * 100).toFixed(1)}% avg confidence</span>
            </div>
          )}
        </div>
        
        <div className="flex items-center gap-1 text-xs text-muted-foreground">
          <Clock className="w-3 h-3" />
          <span>
            Created {new Date(profile.created_timestamp).toLocaleDateString()}
          </span>
        </div>
      </CardContent>
    </Card>
  );
}

type CatProfileFormProps<T> = {
  initialData?: Partial<CatProfile>;
  onSubmit: (data: T) => Promise<void>;
};

function CatProfileForm<T>({ initialData, onSubmit }: CatProfileFormProps<T>) {
  const [formData, setFormData] = useState({
    name: initialData?.name || '',
    description: initialData?.description || '',
    color: initialData?.color || '',
    breed: initialData?.breed || '',
    favorite_activities: initialData?.favorite_activities?.join(', ') || '',
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!formData.name.trim()) {
      setError('Cat name is required');
      return;
    }

    try {
      setIsSubmitting(true);
      setError(null);

      const activities = formData.favorite_activities
        .split(',')
        .map(activity => activity.trim())
        .filter(activity => activity.length > 0);

      const submitData: CreateCatProfileRequest | UpdateCatProfileRequest = {
        name: formData.name.trim(),
      };

      // Only include fields that have values
      if (formData.description.trim()) {
        submitData.description = formData.description.trim();
      }
      if (formData.color.trim()) {
        submitData.color = formData.color.trim();
      }
      if (formData.breed.trim()) {
        submitData.breed = formData.breed.trim();
      }
      if (activities.length > 0) {
        submitData.favorite_activities = activities;
      }

      await onSubmit(submitData as T);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save cat profile');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      {error && (
        <div className="text-sm text-red-500 bg-red-50 border border-red-200 rounded p-3">
          {error}
        </div>
      )}
      
      <div>
        <Label htmlFor="name">Cat Name *</Label>
        <Input
          id="name"
          value={formData.name}
          onChange={(e) => setFormData(prev => ({ ...prev, name: e.target.value }))}
          placeholder="Enter cat name"
          required
        />
      </div>

      <div>
        <Label htmlFor="description">Description</Label>
        <Textarea
          id="description"
          value={formData.description}
          onChange={(e) => setFormData(prev => ({ ...prev, description: e.target.value }))}
          placeholder="Describe the cat's appearance or personality"
          rows={3}
        />
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <Label htmlFor="color">Color/Pattern</Label>
          <Input
            id="color"
            value={formData.color}
            onChange={(e) => setFormData(prev => ({ ...prev, color: e.target.value }))}
            placeholder="e.g., Orange tabby"
          />
        </div>

        <div>
          <Label htmlFor="breed">Breed</Label>
          <Input
            id="breed"
            value={formData.breed}
            onChange={(e) => setFormData(prev => ({ ...prev, breed: e.target.value }))}
            placeholder="e.g., Maine Coon"
          />
        </div>
      </div>

      <div>
        <Label htmlFor="activities">Favorite Activities</Label>
        <Input
          id="activities"
          value={formData.favorite_activities}
          onChange={(e) => setFormData(prev => ({ ...prev, favorite_activities: e.target.value }))}
          placeholder="sleeping, playing, eating (comma-separated)"
        />
        <p className="text-xs text-muted-foreground mt-1">
          Enter activities separated by commas
        </p>
      </div>

      <div className="flex justify-end gap-3 pt-4">
        <Button type="submit" disabled={isSubmitting}>
          {isSubmitting ? 'Saving...' : initialData ? 'Update Profile' : 'Create Profile'}
        </Button>
      </div>
    </form>
  );
}